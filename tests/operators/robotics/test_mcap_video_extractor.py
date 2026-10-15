from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib
import io
from pathlib import Path
import struct
import threading
from types import SimpleNamespace
from typing import Iterator

import av
import daft
import mmops.operators as operators
from mmops.operators.robotics import mcap_to_mp4 as mcap_module
import pyarrow as pa
import pytest
import tos
from mcap.writer import Writer as McapWriter
from mcap_ros2.writer import Writer


COMPRESSED_IMAGE_MSGDEF = """\
std_msgs/Header header
string format
uint8[] data

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


@contextmanager
def _serve_directory(directory: Path) -> Iterator[str]:
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(QuietHandler, directory=str(directory)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _h264_packets(*, b_frames: int = 0) -> list[bytes]:
    codec = av.CodecContext.create("libx264", "w")
    codec.width = 64
    codec.height = 48
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.framerate = Fraction(30, 1)
    codec.options = {
        "preset": "ultrafast",
        "x264-params": (
            "annexb=1:repeat-headers=1:keyint=30:min-keyint=30:scenecut=0:"
            f"bframes={b_frames}:b-adapt=0"
        ),
    }
    if b_frames == 0:
        codec.options["tune"] = "zerolatency"
    codec.open()

    packets: list[bytes] = []
    luma_values = (32, 96, 160) if b_frames == 0 else (32, 64, 96, 128, 160, 192)
    for frame_index, luma in enumerate(luma_values):
        frame = av.VideoFrame(64, 48, "yuv420p")
        frame.pts = frame_index
        for plane_index, plane in enumerate(frame.planes):
            value = luma if plane_index == 0 else 128
            plane.update(bytes([value]) * plane.buffer_size)
        packets.extend(bytes(packet) for packet in codec.encode(frame))
    packets.extend(bytes(packet) for packet in codec.encode(None))
    assert len(packets) == len(luma_values)
    return packets


def _jpeg_packets() -> list[bytes]:
    codec = av.CodecContext.create("mjpeg", "w")
    codec.width = 64
    codec.height = 48
    codec.pix_fmt = "yuvj420p"
    codec.time_base = Fraction(1, 30)
    codec.framerate = Fraction(30, 1)
    codec.open()

    packets: list[bytes] = []
    for frame_index, luma in enumerate((32, 96, 160)):
        frame = av.VideoFrame(64, 48, "yuv420p")
        frame.pts = frame_index
        for plane_index, plane in enumerate(frame.planes):
            value = luma if plane_index == 0 else 128
            plane.update(bytes([value]) * plane.buffer_size)
        packets.extend(bytes(packet) for packet in codec.encode(frame))
    packets.extend(bytes(packet) for packet in codec.encode(None))
    assert len(packets) == 3
    return packets


def _append_cdr_string(payload: bytearray, value: str) -> None:
    encoded = value.encode() + b"\0"
    payload.extend(struct.pack("<I", len(encoded)))
    payload.extend(encoded)
    payload.extend(b"\0" * (-len(payload) % 4))


def _foxglove_compressed_image_cdr(
    *,
    timestamp_ns: int,
    frame_id: str,
    format_: str,
    data: bytes,
) -> bytes:
    payload = bytearray(b"\0\0\0\0")
    payload.extend(
        struct.pack(
            "<iI",
            timestamp_ns // 1_000_000_000,
            timestamp_ns % 1_000_000_000,
        )
    )
    _append_cdr_string(payload, frame_id)
    _append_cdr_string(payload, format_)
    payload.extend(struct.pack("<I", len(data)))
    payload.extend(data)
    payload.extend(b"\0" * (-len(payload) % 4))
    return bytes(payload)


def _write_foxglove_compressed_image_mcap(
    path: Path,
    *,
    topic: str = "ego_video/head",
    packets: list[bytes] | None = None,
) -> list[bytes]:
    source_packets = packets or _jpeg_packets()
    writer = McapWriter(str(path))
    writer.start(profile="foxglove")
    schema_id = writer.register_schema(
        name="foxglove.CompressedImage",
        encoding="jsonschema",
        data=(
            b'{"type":"object","title":"foxglove.CompressedImage",'
            b'"properties":{"timestamp":{"type":"object"},'
            b'"frame_id":{"type":"string"},"format":{"type":"string"},'
            b'"data":{"type":"string","contentEncoding":"base64"}}}'
        ),
    )
    channel_id = writer.register_channel(
        topic=topic,
        message_encoding="cdr",
        schema_id=schema_id,
    )
    start_ns = 1_700_000_000_000_000_000
    for frame_index, packet in enumerate(source_packets):
        timestamp_ns = start_ns + frame_index * 33_333_333
        writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
            sequence=frame_index,
            data=_foxglove_compressed_image_cdr(
                timestamp_ns=timestamp_ns,
                frame_id="ego_video_head",
                format_="jpeg",
                data=packet,
            ),
        )
    writer.finish()
    return source_packets


def _write_compressed_image_mcap(
    path: Path,
    *,
    format_: str = "h264",
    topics: tuple[str, ...] = ("/camera/image/compressed",),
    packets: list[bytes] | None = None,
    topic_formats: dict[str, str] | None = None,
) -> None:
    start_ns = 1_700_000_000_000_000_000
    with Writer(str(path)) as writer:
        schema = writer.register_msgdef(
            "sensor_msgs/msg/CompressedImage",
            COMPRESSED_IMAGE_MSGDEF,
        )
        for frame_index, payload in enumerate(packets or _h264_packets()):
            timestamp_ns = start_ns + frame_index * 33_333_333
            for topic in topics:
                topic_format = (
                    topic_formats[topic] if topic_formats is not None else format_
                )
                message = SimpleNamespace(
                    header=SimpleNamespace(
                        stamp=SimpleNamespace(
                            sec=timestamp_ns // 1_000_000_000,
                            nanosec=timestamp_ns % 1_000_000_000,
                        ),
                        frame_id="camera",
                    ),
                    format=topic_format,
                    data=payload if topic_format == "h264" else b"jpeg-data",
                )
                writer.write_message(
                    topic,
                    schema,
                    message,
                    log_time=timestamp_ns,
                    publish_time=timestamp_ns,
                    sequence=frame_index,
                )


def test_mcap_video_extractor_is_exported_from_robotics_and_root_catalogs() -> None:
    robotics = importlib.import_module("mmops.operators.robotics")

    assert operators.McapVideoExtractor is robotics.McapVideoExtractor
    assert operators.MCAP_VIDEO_RESULT_DTYPE is robotics.MCAP_VIDEO_RESULT_DTYPE
    assert operators.MCAP_VIDEO_RESULT_DTYPE == pa.struct(
        [
            pa.field("output_uri", pa.string(), nullable=False),
            pa.field("topic", pa.string(), nullable=False),
            pa.field("frame_count", pa.int64(), nullable=False),
            pa.field("duration_ns", pa.int64(), nullable=False),
            pa.field("size_bytes", pa.int64(), nullable=False),
            pa.field("sha256", pa.string(), nullable=False),
            pa.field("codec", pa.string(), nullable=False),
        ]
    )


def test_mcap_video_extractor_has_no_legacy_video_catalog_alias() -> None:
    assert importlib.util.find_spec("mmops.operators.video") is None


def test_extractor_handle_exposes_only_the_extract_entrypoint() -> None:
    extractor = operators.McapVideoExtractor()

    assert callable(extractor.extract)
    with pytest.raises(AttributeError, match="_extract_local"):
        getattr(extractor, "_extract_local")


def test_extract_remuxes_local_h264_mcap_without_reencoding(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)

    result = operators.McapVideoExtractor().extract(str(source), str(output))

    assert result == {
        "output_uri": str(output),
        "topic": "/camera/image/compressed",
        "frame_count": 3,
        "duration_ns": 99_999_999,
        "size_bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "codec": "h264",
    }
    with av.open(output) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        duration_ns = int(stream.duration * stream.time_base * 1_000_000_000)
    assert stream.codec_context.name == "h264"
    assert (stream.codec_context.width, stream.codec_context.height) == (64, 48)
    assert len(frames) == 3
    assert duration_ns == result["duration_ns"]


def test_extract_streams_mcap_from_http_input(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)

    with _serve_directory(tmp_path) as base_url:
        result = operators.McapVideoExtractor().extract(
            f"{base_url}/camera.mcap",
            str(output),
        )

    assert result == {
        "output_uri": str(output),
        "topic": "/camera/image/compressed",
        "frame_count": 3,
        "duration_ns": 99_999_999,
        "size_bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "codec": "h264",
    }


@pytest.mark.parametrize("scheme", ["s3", "ftp"])
def test_extract_rejects_unsupported_input_uri_schemes(
    tmp_path: Path,
    scheme: str,
) -> None:
    output = tmp_path / "camera.mp4"

    with pytest.raises(
        ValueError,
        match=rf"mcap_uri.*unsupported URI scheme {scheme!r}",
    ):
        operators.McapVideoExtractor().extract(
            f"{scheme}://input/camera.mcap",
            str(output),
        )

    assert not output.exists()


@pytest.mark.parametrize("scheme", ["http", "https", "s3"])
def test_extract_rejects_non_publishable_output_uri_schemes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        ValueError,
        match=rf"mp4_uri.*unsupported URI scheme {scheme!r}",
    ):
        operators.McapVideoExtractor().extract(
            str(source),
            f"{scheme}://output/camera.mp4",
        )

    assert not (tmp_path / f"{scheme}:").exists()


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_extract_rejects_http_input_without_a_host(
    tmp_path: Path,
    scheme: str,
) -> None:
    output = tmp_path / "camera.mp4"

    with pytest.raises(ValueError, match="invalid HTTP input URI"):
        operators.McapVideoExtractor().extract(
            f"{scheme}:///camera.mcap",
            str(output),
        )

    assert not output.exists()


def test_extract_directly_remuxes_foxglove_jpeg_packets_without_reencoding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    source_packets = _write_foxglove_compressed_image_mcap(source)

    result = operators.McapVideoExtractor(output_codec="source").extract(
        str(source),
        str(output),
    )

    assert result == {
        "output_uri": str(output),
        "topic": "ego_video/head",
        "frame_count": 3,
        "duration_ns": 99_999_999,
        "size_bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "codec": "mjpeg",
    }
    with av.open(output) as container:
        stream = container.streams.video[0]
        output_packets = [
            bytes(packet) for packet in container.demux(stream) if packet.size > 0
        ]
    with av.open(output) as container:
        stream = container.streams.video[0]
        decoded_frames = list(container.decode(stream))
    assert output_packets == source_packets
    assert stream.codec_context.name == "mjpeg"
    assert (stream.codec_context.width, stream.codec_context.height) == (64, 48)
    assert len(decoded_frames) == 3


def test_extract_defaults_to_h264_for_foxglove_jpeg_packets(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    source_packets = _write_foxglove_compressed_image_mcap(source)

    result = operators.McapVideoExtractor(
        h264_crf=23,
        h264_preset="ultrafast",
    ).extract(str(source), str(output))

    with av.open(output) as container:
        stream = container.streams.video[0]
        output_packets = [
            bytes(packet) for packet in container.demux(stream) if packet.size > 0
        ]
    with av.open(output) as container:
        stream = container.streams.video[0]
        decoded_frames = list(container.decode(stream))
        duration_ns = int(stream.duration * stream.time_base * 1_000_000_000)
    assert result["codec"] == "h264"
    assert result["topic"] == "ego_video/head"
    assert result["frame_count"] == 3
    assert result["duration_ns"] == 99_999_999
    assert output_packets != source_packets
    assert stream.codec_context.name == "h264"
    assert (stream.codec_context.width, stream.codec_context.height) == (64, 48)
    assert len(decoded_frames) == 3
    assert duration_ns == result["duration_ns"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"output_codec": "vp9"}, "output_codec"),
        ({"h264_crf": -1}, "h264_crf"),
        ({"h264_crf": 52}, "h264_crf"),
        ({"h264_crf": True}, "h264_crf"),
        ({"h264_preset": "quick"}, "h264_preset"),
    ],
)
def test_extractor_rejects_invalid_video_encoding_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        operators.McapVideoExtractor(**kwargs).extract(None, None)


def test_tos_configuration_is_lazy_for_local_paths(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)

    extractor = operators.McapVideoExtractor(temp_dir=str(tmp_path))

    assert extractor.extract(str(source), str(output))["frame_count"] == 3


def test_extract_streams_from_tos_and_uploads_verified_mp4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    objects = {("input", "camera.mcap"): source.read_bytes()}
    metadata: dict[tuple[str, str], dict[str, str]] = {}
    download = None

    class ReaderAdapterWithoutClose:
        """Matches the CRC adapter returned by tos 2.9.x."""

        def __init__(self, data: bytes) -> None:
            self.data = io.BytesIO(data)

        def read(self, amount: int | None = None) -> bytes:
            return self.data.read(amount)

    class StreamingObject:
        def __init__(self, data: bytes) -> None:
            self.content = ReaderAdapterWithoutClose(data)

        def read(self, amount: int | None = None) -> bytes:
            return self.content.read(amount)

    class FakeTosClient:
        def __init__(self, **_: object) -> None:
            pass

        def get_bucket_version(self, bucket: str) -> object:
            assert bucket == "output"
            return SimpleNamespace(status=SimpleNamespace(value="Unknown"))

        def get_object(self, bucket: str, key: str) -> StreamingObject:
            nonlocal download
            download = StreamingObject(objects[(bucket, key)])
            return download

        def put_object_from_file(
            self,
            bucket: str,
            key: str,
            file_path: str,
            *,
            forbid_overwrite: bool,
            content_type: str,
            meta: dict[str, str],
        ) -> object:
            assert forbid_overwrite is True
            assert content_type == "video/mp4"
            assert bucket == "output"
            assert key.startswith("camera.mp4.mmops-staging-")
            objects[(bucket, key)] = Path(file_path).read_bytes()
            metadata[(bucket, key)] = meta
            return SimpleNamespace(etag=meta["sha256"])

        def head_object(self, bucket: str, key: str) -> object:
            data = objects[(bucket, key)]
            return SimpleNamespace(
                content_length=len(data),
                meta=metadata[(bucket, key)],
                etag=hashlib.sha256(data).hexdigest(),
                version_id=None,
            )

        def copy_object(
            self,
            bucket: str,
            key: str,
            src_bucket: str,
            src_key: str,
            *,
            copy_source_if_match: str,
            forbid_overwrite: bool,
        ) -> object:
            assert (bucket, key) == ("output", "camera.mp4")
            assert forbid_overwrite is True
            assert (
                copy_source_if_match
                == hashlib.sha256(objects[(src_bucket, src_key)]).hexdigest()
            )
            assert (bucket, key) not in objects
            objects[(bucket, key)] = objects[(src_bucket, src_key)]
            metadata[(bucket, key)] = metadata[(src_bucket, src_key)]
            return SimpleNamespace(
                etag=copy_source_if_match,
                version_id=None,
            )

        def delete_object(
            self,
            bucket: str,
            key: str,
            version_id: str | None = None,
        ) -> object:
            assert version_id is None
            objects.pop((bucket, key), None)
            metadata.pop((bucket, key), None)
            return SimpleNamespace()

    monkeypatch.setenv("TOS_ACCESS_KEY", "test-ak")
    monkeypatch.setenv("TOS_SECRET_KEY", "test-sk")
    monkeypatch.setenv("TOS_ENDPOINT", "tos-cn-beijing.volces.com")
    monkeypatch.setenv("TOS_REGION", "cn-beijing")
    monkeypatch.setattr(tos, "TosClientV2", FakeTosClient)

    result = operators.McapVideoExtractor(
        temp_dir=str(tmp_path),
    ).extract(
        "tos://input/camera.mcap",
        "tos://output/camera.mp4",
    )

    assert download is not None
    assert download.content.data.closed
    assert not any("mmops-staging" in key for _, key in objects)
    mp4_bytes = objects[("output", "camera.mp4")]
    extracted = tmp_path / "uploaded.mp4"
    extracted.write_bytes(mp4_bytes)
    with av.open(extracted) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 3
    assert result == {
        "output_uri": "tos://output/camera.mp4",
        "topic": "/camera/image/compressed",
        "frame_count": 3,
        "duration_ns": 99_999_999,
        "size_bytes": len(mp4_bytes),
        "sha256": hashlib.sha256(mp4_bytes).hexdigest(),
        "codec": "h264",
    }


def test_extract_rejects_a_truncated_tos_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    source_bytes = source.read_bytes()
    objects: dict[tuple[str, str], bytes] = {}
    deleted: list[tuple[str, str]] = []

    class StreamingObject:
        def __init__(self, data: bytes) -> None:
            self.content = io.BytesIO(data)

        def read(self, amount: int | None = None) -> bytes:
            return self.content.read(amount)

    class CorruptingTosClient:
        def __init__(self, **_: object) -> None:
            pass

        def get_bucket_version(self, bucket: str) -> object:
            assert bucket == "output"
            return SimpleNamespace(status=SimpleNamespace(value="Unknown"))

        def get_object(self, bucket: str, key: str) -> StreamingObject:
            assert (bucket, key) == ("input", "camera.mcap")
            return StreamingObject(source_bytes)

        def put_object_from_file(
            self,
            bucket: str,
            key: str,
            file_path: str,
            **_: object,
        ) -> object:
            assert bucket == "output"
            assert key.startswith("camera.mp4.mmops-staging-")
            objects[(bucket, key)] = Path(file_path).read_bytes()[:-1]
            return SimpleNamespace(etag="truncated")

        def head_object(self, bucket: str, key: str) -> object:
            data = objects[(bucket, key)]
            return SimpleNamespace(
                content_length=len(data),
                meta={},
                etag="truncated",
                version_id=None,
            )

        def copy_object(self, *_: object, **__: object) -> object:
            pytest.fail("corrupt staging object must not be copied to the final key")

        def delete_object(
            self,
            bucket: str,
            key: str,
            version_id: str | None = None,
        ) -> object:
            assert version_id is None
            deleted.append((bucket, key))
            objects.pop((bucket, key), None)
            return SimpleNamespace()

    monkeypatch.setenv("TOS_ACCESS_KEY", "test-ak")
    monkeypatch.setenv("TOS_SECRET_KEY", "test-sk")
    monkeypatch.setenv("TOS_ENDPOINT", "tos-cn-beijing.volces.com")
    monkeypatch.setenv("TOS_REGION", "cn-beijing")
    monkeypatch.setattr(tos, "TosClientV2", CorruptingTosClient)

    with pytest.raises(ValueError, match="published TOS object size"):
        operators.McapVideoExtractor(temp_dir=str(tmp_path)).extract(
            "tos://input/camera.mcap",
            "tos://output/camera.mp4",
        )

    assert ("output", "camera.mp4") not in objects
    assert len(deleted) == 1
    assert not objects


@pytest.mark.parametrize("status", ["Enabled", "Suspended"])
def test_tos_publication_rejects_a_versioned_bucket(
    tmp_path: Path,
    status: str,
) -> None:
    temporary = tmp_path / "verified.mp4"
    temporary.write_bytes(b"verified")

    class VersionedTosClient:
        def get_bucket_version(self, bucket: str) -> object:
            assert bucket == "output"
            return SimpleNamespace(status=SimpleNamespace(value=status))

        def put_object_from_file(self, *_: object, **__: object) -> object:
            pytest.fail("versioned buckets must be rejected before upload")

    with pytest.raises(ValueError, match=f"versioning is {status}"):
        mcap_module._publish_output(
            temporary,
            "tos://output/camera.mp4",
            VersionedTosClient(),
            hashlib.sha256(temporary.read_bytes()).hexdigest(),
            temporary.stat().st_size,
        )


def test_tos_staging_is_cleaned_when_upload_commits_then_raises(
    tmp_path: Path,
) -> None:
    temporary = tmp_path / "verified.mp4"
    temporary.write_bytes(b"verified")
    objects: dict[tuple[str, str], bytes] = {}

    class CommitThenRaiseTosClient:
        def get_bucket_version(self, bucket: str) -> object:
            assert bucket == "output"
            return SimpleNamespace(status=SimpleNamespace(value="Unknown"))

        def put_object_from_file(
            self,
            bucket: str,
            key: str,
            file_path: str,
            **_: object,
        ) -> object:
            assert key.startswith("camera.mp4.mmops-staging-")
            objects[(bucket, key)] = Path(file_path).read_bytes()
            raise ConnectionError("response lost after commit")

        def head_object(self, bucket: str, key: str) -> object:
            assert (bucket, key) in objects
            return SimpleNamespace(version_id=None)

        def delete_object(
            self,
            bucket: str,
            key: str,
            version_id: str | None = None,
        ) -> object:
            assert version_id is None
            objects.pop((bucket, key), None)
            return SimpleNamespace()

    with pytest.raises(ConnectionError, match="response lost after commit"):
        mcap_module._publish_output(
            temporary,
            "tos://output/camera.mp4",
            CommitThenRaiseTosClient(),
            hashlib.sha256(temporary.read_bytes()).hexdigest(),
            temporary.stat().st_size,
        )

    assert not objects


def test_tos_publication_removes_exact_version_if_versioning_changes_midflight(
    tmp_path: Path,
) -> None:
    temporary = tmp_path / "verified.mp4"
    temporary.write_bytes(b"verified")
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    deleted: list[tuple[str, str, str | None]] = []

    class VersioningRaceTosClient:
        def get_bucket_version(self, bucket: str) -> object:
            return SimpleNamespace(status=SimpleNamespace(value="Unknown"))

        def put_object_from_file(
            self,
            bucket: str,
            key: str,
            file_path: str,
            **_: object,
        ) -> object:
            return SimpleNamespace(etag=digest, version_id=None)

        def head_object(self, bucket: str, key: str) -> object:
            return SimpleNamespace(
                content_length=temporary.stat().st_size,
                meta={"sha256": digest},
                etag=digest,
                version_id=None,
            )

        def copy_object(self, *_: object, **__: object) -> object:
            return SimpleNamespace(etag=digest, version_id="new-version")

        def delete_object(
            self,
            bucket: str,
            key: str,
            version_id: str | None = None,
        ) -> object:
            deleted.append((bucket, key, version_id))
            return SimpleNamespace()

    with pytest.raises(RuntimeError, match="versioning changed"):
        mcap_module._publish_output(
            temporary,
            "tos://output/camera.mp4",
            VersioningRaceTosClient(),
            digest,
            temporary.stat().st_size,
        )

    assert ("output", "camera.mp4", "new-version") in deleted
    assert any("mmops-staging" in key for _, key, _ in deleted)


def test_tos_publication_never_deletes_an_ambiguous_unversioned_final_key(
    tmp_path: Path,
) -> None:
    temporary = tmp_path / "verified.mp4"
    temporary.write_bytes(b"verified")
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    deleted: list[str] = []
    head_calls = 0

    class AmbiguousFinalTosClient:
        def get_bucket_version(self, bucket: str) -> object:
            return SimpleNamespace(status=SimpleNamespace(value="Unknown"))

        def put_object_from_file(
            self,
            bucket: str,
            key: str,
            file_path: str,
            **_: object,
        ) -> object:
            return SimpleNamespace(etag=digest, version_id=None)

        def head_object(self, bucket: str, key: str) -> object:
            nonlocal head_calls
            head_calls += 1
            if "mmops-staging" in key:
                return SimpleNamespace(
                    content_length=temporary.stat().st_size,
                    meta={"sha256": digest},
                    etag=digest,
                    version_id=None,
                )
            return SimpleNamespace(
                content_length=1,
                meta={},
                etag="concurrent-writer",
                version_id=None,
            )

        def copy_object(self, *_: object, **__: object) -> object:
            return SimpleNamespace(etag=digest, version_id=None)

        def delete_object(
            self,
            bucket: str,
            key: str,
            version_id: str | None = None,
        ) -> object:
            deleted.append(key)
            return SimpleNamespace()

    with pytest.raises(RuntimeError, match="preserved.*concurrent writes"):
        mcap_module._publish_output(
            temporary,
            "tos://output/camera.mp4",
            AmbiguousFinalTosClient(),
            digest,
            temporary.stat().st_size,
        )

    assert head_calls >= 2
    assert "camera.mp4" not in deleted
    assert any("mmops-staging" in key for key in deleted)


def test_extract_runs_in_a_real_daft_dataframe(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)
    df = daft.from_pydict(
        {
            "mcap_uri": [str(source)],
            "mp4_uri": [str(output)],
        }
    )

    result = df.with_column(
        "video",
        operators.McapVideoExtractor().extract(df["mcap_uri"], df["mp4_uri"]),
    ).to_pydict()

    assert result["video"] == [
        {
            "output_uri": str(output),
            "topic": "/camera/image/compressed",
            "frame_count": 3,
            "duration_ns": 99_999_999,
            "size_bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "codec": "h264",
        }
    ]


def test_jpeg_to_h264_encoding_runs_in_a_real_daft_dataframe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_foxglove_compressed_image_mcap(source)
    df = daft.from_pydict(
        {
            "mcap_uri": [str(source)],
            "mp4_uri": [str(output)],
        }
    )

    result = df.with_column(
        "video",
        operators.McapVideoExtractor(
            output_codec="h264",
            h264_crf=23,
            h264_preset="ultrafast",
        ).extract(df["mcap_uri"], df["mp4_uri"]),
    ).to_pydict()["video"][0]

    with av.open(output) as container:
        stream = container.streams.video[0]
        decoded_frames = list(container.decode(stream))
    assert result["codec"] == "h264"
    assert result["frame_count"] == 3
    assert stream.codec_context.name == "h264"
    assert len(decoded_frames) == 3


def test_extract_requires_topic_when_mcap_has_multiple_h264_streams(
    tmp_path: Path,
) -> None:
    source = tmp_path / "multi-camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(
        source,
        topics=("/left/image/compressed", "/right/image/compressed"),
    )

    with pytest.raises(ValueError, match="multiple.*pass topic explicitly"):
        operators.McapVideoExtractor().extract(str(source), str(output))

    assert not output.exists()
    result = operators.McapVideoExtractor().extract(
        str(source),
        str(output),
        topic="/right/image/compressed",
    )
    assert result["topic"] == "/right/image/compressed"
    assert result["frame_count"] == 3


def test_extract_rejects_non_h264_compressed_images_without_publishing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "jpeg.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source, format_="jpeg")

    with pytest.raises(ValueError, match="format 'jpeg'.*expected 'h264'"):
        operators.McapVideoExtractor().extract(str(source), str(output))

    assert not output.exists()


def test_extract_auto_selects_the_only_h264_topic_from_mixed_formats(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed-formats.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(
        source,
        topics=("/preview/jpeg", "/camera/image/compressed"),
        topic_formats={
            "/preview/jpeg": "jpeg",
            "/camera/image/compressed": "h264",
        },
    )

    result = operators.McapVideoExtractor().extract(str(source), str(output))

    assert result["topic"] == "/camera/image/compressed"
    assert result["frame_count"] == 3


def test_extract_rejects_undecodable_h264_without_publishing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid-h264.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(
        source,
        packets=[b"not-h264-1", b"not-h264-2", b"not-h264-3"],
    )

    with pytest.raises((ValueError, av.error.InvalidDataError)):
        operators.McapVideoExtractor().extract(str(source), str(output))

    assert not output.exists()


def test_extract_rejects_h264_with_b_frame_reordering(tmp_path: Path) -> None:
    source = tmp_path / "b-frames.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source, packets=_h264_packets(b_frames=2))

    with pytest.raises(ValueError, match="B-frame"):
        operators.McapVideoExtractor().extract(str(source), str(output))

    assert not output.exists()


def test_extract_refuses_to_overwrite_an_existing_local_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)
    output.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError, match="output already exists"):
        operators.McapVideoExtractor().extract(str(source), str(output))

    assert output.read_bytes() == b"keep-me"


def test_extract_preserves_null_rows_and_rejects_partial_nulls() -> None:
    extractor = operators.McapVideoExtractor()

    assert extractor.extract(None, None) is None
    with pytest.raises(ValueError, match="both be null or both be set"):
        extractor.extract(None, "/tmp/output.mp4")
