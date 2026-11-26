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
import fsspec
import mmops.operators as operators
from mmops.operators.robotics import mcap_to_mp4 as mcap_module
import pyarrow as pa
import pytest
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


def _jpeg_packet(width: int, height: int) -> bytes:
    codec = av.CodecContext.create("mjpeg", "w")
    codec.width = width
    codec.height = height
    codec.pix_fmt = "yuvj420p"
    codec.time_base = Fraction(1, 30)
    codec.open()
    frame = av.VideoFrame(width, height, "yuv420p")
    for plane_index, plane in enumerate(frame.planes):
        plane.update(bytes([96 if plane_index == 0 else 128]) * plane.buffer_size)
    packets = list(codec.encode(frame)) + list(codec.encode(None))
    assert len(packets) == 1
    return bytes(packets[0])


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
    format_: str = "jpeg",
    packets: list[bytes] | None = None,
    trailing_attachment: bytes | None = None,
) -> list[bytes]:
    source_packets = packets or _jpeg_packets()
    writer = McapWriter(
        str(path),
        chunk_size=256 if trailing_attachment is not None else 1_048_576,
    )
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
                format_=format_,
                data=packet,
            ),
        )
    if trailing_attachment is not None:
        writer.add_attachment(
            create_time=start_ns,
            log_time=start_ns,
            name="unrelated.bin",
            media_type="application/octet-stream",
            data=trailing_attachment,
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
    frame_formats: tuple[str, ...] | None = None,
) -> None:
    start_ns = 1_700_000_000_000_000_000
    source_packets = packets or _h264_packets()
    if frame_formats is not None and len(frame_formats) != len(source_packets):
        raise ValueError("frame_formats must match the packet count")
    with Writer(str(path)) as writer:
        schema = writer.register_msgdef(
            "sensor_msgs/msg/CompressedImage",
            COMPRESSED_IMAGE_MSGDEF,
        )
        for frame_index, payload in enumerate(source_packets):
            timestamp_ns = start_ns + frame_index * 33_333_333
            for topic in topics:
                topic_format = (
                    frame_formats[frame_index]
                    if frame_formats is not None
                    else topic_formats[topic] if topic_formats is not None else format_
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
                    data=payload,
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


def test_extract_uses_ros2_format_to_encode_jpeg_as_h264(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(
        source,
        format_="jpeg",
        packets=_jpeg_packets(),
    )

    result = operators.McapVideoExtractor(
        h264_preset="ultrafast",
    ).extract(str(source), str(output))

    with av.open(output) as container:
        stream = container.streams.video[0]
        decoded_frames = list(container.decode(stream))
        duration_ns = int(stream.duration * stream.time_base * 1_000_000_000)
    assert result["codec"] == "h264"
    assert result["topic"] == "/camera/image/compressed"
    assert result["frame_count"] == 3
    assert stream.codec_context.name == "h264"
    assert (stream.codec_context.width, stream.codec_context.height) == (64, 48)
    assert len(decoded_frames) == 3
    assert duration_ns == result["duration_ns"]


def test_extract_uses_foxglove_format_to_remux_h264(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_foxglove_compressed_image_mcap(
        source,
        format_="h264",
        packets=_h264_packets(),
    )

    result = operators.McapVideoExtractor().extract(str(source), str(output))

    with av.open(output) as container:
        stream = container.streams.video[0]
        decoded_frames = list(container.decode(stream))
        duration_ns = int(stream.duration * stream.time_base * 1_000_000_000)
    assert result["codec"] == "h264"
    assert result["topic"] == "ego_video/head"
    assert result["frame_count"] == 3
    assert stream.codec_context.name == "h264"
    assert (stream.codec_context.width, stream.codec_context.height) == (64, 48)
    assert len(decoded_frames) == 3
    assert duration_ns == result["duration_ns"]


def test_extract_validates_mp4_with_one_demux_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)
    original_open = av.open
    validation_opens = 0
    validation_streams: list[object] = []

    def tracked_open(*args: object, **kwargs: object) -> object:
        nonlocal validation_opens
        if kwargs.get("mode", "r") == "r":
            validation_opens += 1
        container = original_open(*args, **kwargs)
        if kwargs.get("mode", "r") == "r":
            validation_streams.append(container.streams.video[0])
        return container

    monkeypatch.setattr(av, "open", tracked_open)

    result = operators.McapVideoExtractor().extract(str(source), str(output))

    assert result["frame_count"] == 3
    assert validation_opens == 1
    assert len(validation_streams) == 1
    assert validation_streams[0].thread_type.name == "AUTO"


def test_file_uris_use_default_local_filesystem(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)

    result = operators.McapVideoExtractor().extract(source.as_uri(), output.as_uri())

    assert result["output_uri"] == output.as_uri()
    assert result["frame_count"] == 3
    assert output.exists()


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


def test_http_input_uses_an_explicit_input_filesystem(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)
    payload = source.read_bytes()

    class MemoryHttpFilesystem:
        protocol = ("http", "https")

        def open(self, path: str, mode: str, **kwargs: object) -> io.BytesIO:
            assert path == "https://example.test/camera.mcap"
            assert mode == "rb"
            assert kwargs["cache_type"] == "none"
            return io.BytesIO(payload)

    result = operators.McapVideoExtractor(
        input_filesystem=MemoryHttpFilesystem()
    ).extract("https://example.test/camera.mcap", str(output))

    assert result["frame_count"] == 3
    assert output.exists()


def test_nonseeking_mcap_yields_video_before_trailing_attachment(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera-with-attachment.mcap"
    _write_foxglove_compressed_image_mcap(
        source,
        trailing_attachment=b"x" * (2 * 1024 * 1024),
    )

    with source.open("rb") as raw_stream:
        stream = mcap_module._NonSeekingStream(raw_stream)
        first = next(mcap_module._iter_compressed_frames(stream, None, []))
        assert first.topic == "ego_video/head"
        assert raw_stream.tell() < source.stat().st_size // 2


def test_remote_input_coalesces_small_mcap_reads(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)
    payload = source.read_bytes()

    class CountingStream(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(payload)
            self.read_sizes: list[int] = []

        def read(self, size: int = -1) -> bytes:
            self.read_sizes.append(size)
            return super().read(size)

    class CountingFilesystem:
        protocol = "s3"

        def __init__(self) -> None:
            self.stream: CountingStream | None = None

        def open(self, path: str, mode: str, **kwargs: object) -> CountingStream:
            assert path == "input/camera.mcap"
            assert mode == "rb"
            assert kwargs["cache_type"] == "none"
            self.stream = CountingStream()
            return self.stream

    filesystem = CountingFilesystem()
    result = operators.McapVideoExtractor(input_filesystem=filesystem).extract(
        "s3://input/camera.mcap", str(output)
    )

    assert result["frame_count"] == 3
    assert filesystem.stream is not None
    assert filesystem.stream.closed
    assert len(filesystem.stream.read_sizes) <= 3
    assert filesystem.stream.read_sizes[0] >= 256 * 1024


def test_explicit_remote_topic_uses_mcap_index_to_skip_attachment(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera-with-attachment.mcap"
    output = tmp_path / "camera.mp4"
    _write_foxglove_compressed_image_mcap(
        source,
        trailing_attachment=b"x" * (8 * 1024 * 1024),
    )
    payload = source.read_bytes()

    class CountingStream(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(payload)
            self.bytes_read = 0
            self.seek_calls = 0

        def read(self, size: int = -1) -> bytes:
            data = super().read(size)
            self.bytes_read += len(data)
            return data

        def seek(self, offset: int, whence: int = 0) -> int:
            self.seek_calls += 1
            return super().seek(offset, whence)

    class CountingFilesystem:
        protocol = "s3"

        def __init__(self) -> None:
            self.stream: CountingStream | None = None

        def open(self, path: str, mode: str, **kwargs: object) -> CountingStream:
            assert path == "input/camera.mcap"
            assert mode == "rb"
            assert kwargs["cache_type"] == "none"
            self.stream = CountingStream()
            return self.stream

    filesystem = CountingFilesystem()
    result = operators.McapVideoExtractor(input_filesystem=filesystem).extract(
        "s3://input/camera.mcap",
        str(output),
        topic="ego_video/head",
    )

    assert result["frame_count"] == 3
    assert filesystem.stream is not None
    assert filesystem.stream.seek_calls > 0
    assert filesystem.stream.bytes_read < len(payload) // 2


def test_unsupported_formats_keep_only_one_diagnostic(tmp_path: Path) -> None:
    source = tmp_path / "unsupported.mcap"
    _write_compressed_image_mcap(source, format_="png")
    unsupported_formats: list[tuple[str, str]] = []

    with source.open("rb") as stream:
        frames = list(
            mcap_module._iter_compressed_frames(stream, None, unsupported_formats)
        )

    assert frames == []
    assert unsupported_formats == [("/camera/image/compressed", "png")]


@pytest.mark.parametrize("scheme", ["s3", "ftp"])
def test_remote_input_requires_an_injected_filesystem(
    tmp_path: Path,
    scheme: str,
) -> None:
    output = tmp_path / "camera.mp4"

    with pytest.raises(
        ValueError,
        match=rf"input_filesystem.*{scheme}",
    ):
        operators.McapVideoExtractor().extract(
            f"{scheme}://input/camera.mcap",
            str(output),
        )

    assert not output.exists()


@pytest.mark.parametrize("scheme", ["http", "https", "s3"])
def test_remote_output_requires_an_injected_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        ValueError,
        match=rf"output_filesystem.*{scheme}",
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

    with pytest.raises(ValueError, match="not a valid http"):
        operators.McapVideoExtractor().extract(
            f"{scheme}:///camera.mcap",
            str(output),
        )

    assert not output.exists()


@pytest.mark.parametrize("in_source", [True, False])
def test_invalid_tos_uri_error_redacts_query_token(
    tmp_path: Path,
    in_source: bool,
) -> None:
    token_uri = "tos://bucket/camera.mcap?token=SECRET_SENTINEL"
    source = token_uri if in_source else str(tmp_path / "camera.mcap")
    destination = str(tmp_path / "camera.mp4") if in_source else token_uri

    with pytest.raises(ValueError) as caught:
        operators.McapVideoExtractor().extract(source, destination)

    assert "SECRET_SENTINEL" not in str(caught.value)


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


def test_source_mjpeg_decodes_only_the_first_frame_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_foxglove_compressed_image_mcap(source)
    original_decode = mcap_module._decode_single_jpeg
    decode_calls = 0

    def tracked_decode(decoder: object, payload: bytes) -> object:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(decoder, payload)

    monkeypatch.setattr(mcap_module, "_decode_single_jpeg", tracked_decode)

    result = operators.McapVideoExtractor(output_codec="source").extract(
        str(source), str(output)
    )

    assert result["frame_count"] == 3
    assert decode_calls == 1


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


@pytest.mark.parametrize("output_codec", ["h264", "source"])
def test_extract_rejects_jpeg_resolution_change_without_publishing(
    tmp_path: Path,
    output_codec: str,
) -> None:
    source = tmp_path / "changing-resolution.mcap"
    output = tmp_path / "camera.mp4"
    _write_foxglove_compressed_image_mcap(
        source,
        packets=[_jpeg_packets()[0], _jpeg_packet(80, 48)],
    )

    with pytest.raises(ValueError, match="changes JPEG frame dimensions"):
        operators.McapVideoExtractor(output_codec=output_codec).extract(
            str(source), str(output)
        )

    assert not output.exists()


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


def test_local_paths_use_default_filesystems(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)

    extractor = operators.McapVideoExtractor(temp_dir=str(tmp_path))

    assert extractor.extract(str(source), str(output))["frame_count"] == 3


def test_local_paths_use_an_injected_fsspec_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)
    filesystem = fsspec.filesystem("file", skip_instance_cache=True)
    opened: list[tuple[str, str]] = []
    linked: list[tuple[str, str]] = []
    original_open = filesystem.open
    original_link = filesystem.link

    def tracked_open(path: str, mode: str, **kwargs: object) -> object:
        opened.append((str(path), mode))
        return original_open(path, mode, **kwargs)

    def tracked_link(source_path: str, destination_path: str) -> None:
        linked.append((str(source_path), str(destination_path)))
        original_link(source_path, destination_path)

    monkeypatch.setattr(filesystem, "open", tracked_open)
    monkeypatch.setattr(filesystem, "link", tracked_link)

    result = operators.McapVideoExtractor(
        input_filesystem=filesystem, output_filesystem=filesystem
    ).extract(str(source), str(output))

    assert result["frame_count"] == 3
    assert (str(source), "rb") in opened
    assert len(linked) == 1
    assert linked[0][1] == str(output)


def test_incompatible_injected_filesystem_is_not_ignored_for_tos(
    tmp_path: Path,
) -> None:
    extractor = operators.McapVideoExtractor(input_filesystem=fsspec.filesystem("file"))

    with pytest.raises(ValueError, match="filesystem.*tos"):
        extractor.extract("tos://input/camera.mcap", str(tmp_path / "camera.mp4"))


def test_tos_uri_requires_an_injected_filesystem_even_with_worker_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TOS_ENDPOINT",
        "TOS_REGION",
        "TOS_ACCESS_KEY",
        "TOS_SECRET_KEY",
    ):
        monkeypatch.setenv(name, "configured-on-worker")

    def unexpected_filesystem(protocol: str, **_kwargs: object) -> object:
        if protocol == "tos":
            pytest.fail("the operator must not construct a TOS filesystem")
        return object()

    monkeypatch.setattr(fsspec, "filesystem", unexpected_filesystem)
    with pytest.raises(ValueError, match="filesystem.*tos"):
        operators.McapVideoExtractor().extract(
            "tos://input/camera.mcap", str(tmp_path / "camera.mp4")
        )


def test_tos_output_requires_its_own_filesystem_before_reading_input(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="output_filesystem.*tos"):
        operators.McapVideoExtractor().extract(
            str(tmp_path / "missing.mcap"), "tos://output/camera.mp4"
        )


def test_output_filesystem_protocol_is_checked_independently(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="output_filesystem.*tos"):
        operators.McapVideoExtractor(
            output_filesystem=fsspec.filesystem("file")
        ).extract(str(tmp_path / "missing.mcap"), "tos://output/camera.mp4")


def test_remote_output_requires_safe_publisher_before_reading_input() -> None:
    class ReadMustNotStart:
        protocol = "tos"

        def open(self, *_args: object, **_kwargs: object) -> object:
            pytest.fail("input must not be read before validating remote publication")

    with pytest.raises(RuntimeError, match="safe no-overwrite publication support"):
        operators.McapVideoExtractor(
            input_filesystem=ReadMustNotStart(),
            output_filesystem=ReadMustNotStart(),
        ).extract("tos://input/camera.mcap", "tos://output/camera.mp4")


def test_plain_fsspec_remote_output_fails_closed_before_reading_input(
    tmp_path: Path,
) -> None:
    filesystem = fsspec.filesystem("memory", skip_instance_cache=True)

    with pytest.raises(RuntimeError, match="safe no-overwrite publication support"):
        operators.McapVideoExtractor(output_filesystem=filesystem).extract(
            str(tmp_path / "missing.mcap"), "memory://output/camera.mp4"
        )


def test_local_output_requires_hard_link_before_reading_input(tmp_path: Path) -> None:
    class ReadMustNotStart:
        protocol = "file"

        def open(self, *_args: object, **_kwargs: object) -> object:
            pytest.fail("input must not be read before validating local publication")

    with pytest.raises(RuntimeError, match="hard-link publication support"):
        operators.McapVideoExtractor(
            input_filesystem=ReadMustNotStart(),
            output_filesystem=ReadMustNotStart(),
        ).extract(str(tmp_path / "missing.mcap"), str(tmp_path / "camera.mp4"))


def test_incompatible_injected_filesystem_is_not_ignored_for_local_paths(
    tmp_path: Path,
) -> None:
    class OtherFilesystem:
        protocol = "other"

    extractor = operators.McapVideoExtractor(input_filesystem=OtherFilesystem())

    with pytest.raises(ValueError, match="filesystem.*file"):
        extractor.extract(str(tmp_path / "camera.mcap"), str(tmp_path / "camera.mp4"))


def test_extract_reads_tos_uri_through_fsspec(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)
    payload = source.read_bytes()

    class MemoryTosFilesystem:
        protocol = "tos"

        def open(self, path: str, mode: str, **kwargs: object) -> io.BytesIO:
            assert path == "input/camera.mcap"
            assert mode == "rb"
            return io.BytesIO(payload)

    result = operators.McapVideoExtractor(
        input_filesystem=MemoryTosFilesystem()
    ).extract("tos://input/camera.mcap", str(output))

    assert result["frame_count"] == 3
    assert output.exists()


@pytest.mark.parametrize("protocol", ["s3", "oss"])
def test_extract_reads_any_injected_object_filesystem(
    tmp_path: Path, protocol: str
) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)
    payload = source.read_bytes()

    class MemoryObjectFilesystem:
        def __init__(self) -> None:
            self.protocol = protocol

        def open(self, path: str, mode: str, **kwargs: object) -> io.BytesIO:
            assert path == "input/camera.mcap"
            assert mode == "rb"
            assert kwargs["cache_type"] == "none"
            return io.BytesIO(payload)

    result = operators.McapVideoExtractor(
        input_filesystem=MemoryObjectFilesystem()
    ).extract(f"{protocol}://input/camera.mcap", str(output))

    assert result["frame_count"] == 3
    assert output.exists()


@pytest.mark.parametrize("topic", [None, "/camera/image/compressed"])
def test_extract_reads_tos_stream_that_overreturns_requested_bytes(
    tmp_path: Path, topic: str | None
) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)
    payload = source.read_bytes()

    class OverreadingStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(size + 1 if size >= 0 else size)

    class OverreadingFilesystem:
        protocol = "tos"

        def open(self, path: str, mode: str, **kwargs: object) -> OverreadingStream:
            assert path == "input/camera.mcap"
            assert mode == "rb"
            return OverreadingStream(payload)

    result = operators.McapVideoExtractor(
        input_filesystem=OverreadingFilesystem()
    ).extract("tos://input/camera.mcap", str(output), topic=topic)

    assert result["frame_count"] == 3
    assert output.exists()


class _TrackingBytesIO(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


class _MemoryRemoteFilesystem:
    def __init__(
        self,
        protocol: str,
        objects: dict[str, bytes] | None = None,
        *,
        preflight_error: Exception | None = None,
        corrupt_publication: str | None = None,
        raise_after_publication: Exception | None = None,
        require_final_readback: bool = False,
    ) -> None:
        self.protocol = protocol
        self.objects = objects if objects is not None else {}
        self.preflight_error = preflight_error
        self.corrupt_publication = corrupt_publication
        self.raise_after_publication = raise_after_publication
        self.require_final_readback = require_final_readback
        self.opened: list[_TrackingBytesIO] = []
        self.preflighted: list[str] = []
        self.published: list[str] = []
        self.invalidated: list[str] = []

    def open(self, path: str, mode: str, **kwargs: object) -> _TrackingBytesIO:
        assert mode == "rb"
        assert kwargs.get("cache_type") == "none"
        stream = _TrackingBytesIO(self.objects[path])
        self.opened.append(stream)
        return stream

    def exists(self, path: str) -> bool:
        return path in self.objects

    def info(self, path: str) -> dict[str, int]:
        return {"size": len(self.objects[path])}

    def invalidate_cache(self, path: str) -> None:
        self.invalidated.append(path)

    def preflight_publish(self, path: str) -> None:
        self.preflighted.append(path)
        if self.preflight_error is not None:
            raise self.preflight_error

    def publish_verified_file(
        self, local_path: str, path: str, *, size_bytes: int
    ) -> dict[str, object]:
        assert path not in self.objects
        data = Path(local_path).read_bytes()
        assert len(data) == size_bytes
        sha256 = hashlib.sha256(data).hexdigest()
        self.published.append(path)
        if self.corrupt_publication == "truncate":
            data = data[:-1]
        elif self.corrupt_publication == "change":
            data = bytes([data[0] ^ 1]) + data[1:]
        self.objects[path] = data
        if self.raise_after_publication is not None:
            raise self.raise_after_publication
        verification = (
            "readback_required"
            if self.require_final_readback or self.corrupt_publication is not None
            else "sha256"
        )
        return {
            "size_bytes": size_bytes,
            "sha256": sha256,
            "final_verification": verification,
        }


def test_distinct_remote_input_and_output_filesystems_are_used(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    input_objects = {"input/camera.mcap": source.read_bytes()}
    output_objects: dict[str, bytes] = {}
    input_filesystem = _MemoryRemoteFilesystem("tos", input_objects)
    output_filesystem = _MemoryRemoteFilesystem("oss", output_objects)

    result = operators.McapVideoExtractor(
        input_filesystem=input_filesystem,
        output_filesystem=output_filesystem,
        temp_dir=str(tmp_path),
    ).extract("tos://input/camera.mcap", "oss://output/camera.mp4")

    assert result["frame_count"] == 3
    assert input_objects == {"input/camera.mcap": source.read_bytes()}
    assert "output/camera.mp4" in output_objects
    assert input_filesystem.opened
    assert output_filesystem.opened == []
    assert output_filesystem.preflighted == ["output/camera.mp4"]
    assert output_filesystem.published == ["output/camera.mp4"]


def test_remote_output_preflight_rejects_unsafe_backend_before_reading_input() -> None:
    filesystem = _MemoryRemoteFilesystem(
        "s3", preflight_error=ValueError("create-only copy unavailable")
    )
    with pytest.raises(ValueError, match="create-only copy unavailable"):
        operators.McapVideoExtractor(
            input_filesystem=filesystem, output_filesystem=filesystem
        ).extract("s3://input/missing.mcap", "s3://output/camera.mp4")
    assert filesystem.opened == []
    assert filesystem.published == []


def test_remote_output_preflight_rejects_existing_key_before_reading_input() -> None:
    objects = {"output/camera.mp4": b"existing"}
    filesystem = _MemoryRemoteFilesystem("s3", objects)
    with pytest.raises(FileExistsError, match="already exists"):
        operators.McapVideoExtractor(
            input_filesystem=filesystem, output_filesystem=filesystem
        ).extract("s3://input/missing.mcap", "s3://output/camera.mp4")
    assert filesystem.opened == []
    assert filesystem.published == []
    assert objects["output/camera.mp4"] == b"existing"


@pytest.mark.parametrize("protocol", ["tos", "oss", "s3"])
def test_extract_publishes_verified_mp4_through_remote_filesystem(
    tmp_path: Path, protocol: str
) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    filesystem = _MemoryRemoteFilesystem(protocol)

    result = operators.McapVideoExtractor(
        output_filesystem=filesystem, temp_dir=str(tmp_path)
    ).extract(str(source), f"{protocol}://output/camera.mp4")

    assert filesystem.preflighted == ["output/camera.mp4"]
    assert filesystem.published == ["output/camera.mp4"]
    assert filesystem.invalidated == ["output/camera.mp4"]
    assert filesystem.opened == []
    mp4_bytes = filesystem.objects["output/camera.mp4"]
    extracted = tmp_path / "uploaded.mp4"
    extracted.write_bytes(mp4_bytes)
    with av.open(extracted) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 3
    assert result == {
        "output_uri": f"{protocol}://output/camera.mp4",
        "topic": "/camera/image/compressed",
        "frame_count": 3,
        "duration_ns": 99_999_999,
        "size_bytes": len(mp4_bytes),
        "sha256": hashlib.sha256(mp4_bytes).hexdigest(),
        "codec": "h264",
    }


def test_remote_publisher_digest_avoids_local_rehash_and_final_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    filesystem = _MemoryRemoteFilesystem("s3")

    def unexpected_local_hash(_path: Path) -> str:
        pytest.fail("remote publication must hash while reading the upload")

    monkeypatch.setattr(mcap_module, "_sha256", unexpected_local_hash)

    result = operators.McapVideoExtractor(
        output_filesystem=filesystem, temp_dir=str(tmp_path)
    ).extract(str(source), "s3://output/camera.mp4")

    assert (
        result["sha256"]
        == hashlib.sha256(filesystem.objects["output/camera.mp4"]).hexdigest()
    )
    assert filesystem.opened == []


def test_remote_publisher_can_request_final_readback(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    filesystem = _MemoryRemoteFilesystem("s3", require_final_readback=True)

    result = operators.McapVideoExtractor(
        output_filesystem=filesystem, temp_dir=str(tmp_path)
    ).extract(str(source), "s3://output/camera.mp4")

    assert result["frame_count"] == 3
    assert len(filesystem.opened) == 1
    assert filesystem.opened[0].closed
    assert filesystem.opened[0].read_sizes == [result["size_bytes"] + 1, 1]


def test_remote_publisher_rejects_an_unverified_receipt(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)

    class InvalidReceiptFilesystem(_MemoryRemoteFilesystem):
        def publish_verified_file(
            self, local_path: str, path: str, *, size_bytes: int
        ) -> dict[str, object]:
            receipt = super().publish_verified_file(
                local_path, path, size_bytes=size_bytes
            )
            receipt["final_verification"] = "etag"
            return receipt

    filesystem = InvalidReceiptFilesystem("s3")

    with pytest.raises(
        RuntimeError, match="remote output verification failed"
    ) as error:
        operators.McapVideoExtractor(
            output_filesystem=filesystem, temp_dir=str(tmp_path)
        ).extract(str(source), "s3://output/camera.mp4")

    assert "final_verification" in str(error.value.__cause__)
    assert "output/camera.mp4" in filesystem.objects


def test_remote_readback_is_bounded_if_object_grows_after_info(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)

    class GrowingFilesystem(_MemoryRemoteFilesystem):
        def publish_verified_file(
            self, local_path: str, path: str, *, size_bytes: int
        ) -> dict[str, object]:
            receipt = super().publish_verified_file(
                local_path, path, size_bytes=size_bytes
            )
            self.objects[path] += b"x"
            return receipt

        def info(self, path: str) -> dict[str, int]:
            return {"size": len(self.objects[path]) - 1}

    filesystem = GrowingFilesystem("s3", require_final_readback=True)

    with pytest.raises(
        RuntimeError, match="remote output verification failed"
    ) as error:
        operators.McapVideoExtractor(
            output_filesystem=filesystem, temp_dir=str(tmp_path)
        ).extract(str(source), "s3://output/camera.mp4")

    assert "published object streamed size" in str(error.value.__cause__)
    assert len(filesystem.opened) == 1
    assert filesystem.opened[0].read_sizes == [
        len(filesystem.objects["output/camera.mp4"])
    ]


@pytest.mark.parametrize(
    ("corruption", "cause"),
    [("truncate", "published object size"), ("change", "published object sha256")],
)
def test_extract_rejects_a_corrupted_remote_publication(
    tmp_path: Path, corruption: str, cause: str
) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    filesystem = _MemoryRemoteFilesystem("s3", corrupt_publication=corruption)

    with pytest.raises(
        RuntimeError, match="remote output verification failed"
    ) as error:
        operators.McapVideoExtractor(
            output_filesystem=filesystem, temp_dir=str(tmp_path)
        ).extract(str(source), "s3://output/camera.mp4")

    assert cause in str(error.value.__cause__)
    assert "output/camera.mp4" in filesystem.objects


def test_remote_publication_error_preserves_ambiguous_final_key(tmp_path: Path) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    filesystem = _MemoryRemoteFilesystem(
        "oss", raise_after_publication=ConnectionError("response lost")
    )

    with pytest.raises(ConnectionError, match="response lost"):
        operators.McapVideoExtractor(
            output_filesystem=filesystem, temp_dir=str(tmp_path)
        ).extract(str(source), "oss://output/camera.mp4")

    assert "output/camera.mp4" in filesystem.objects


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


def test_injected_remote_input_runs_in_a_real_daft_dataframe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source)
    filesystem = _MemoryRemoteFilesystem(
        "s3", {"input/camera.mcap": source.read_bytes()}
    )

    df = daft.from_pydict(
        {
            "mcap_uri": ["s3://input/camera.mcap"],
            "mp4_uri": [str(output)],
        }
    )
    result = df.with_column(
        "video",
        operators.McapVideoExtractor(input_filesystem=filesystem).extract(
            df["mcap_uri"], df["mp4_uri"]
        ),
    ).to_pydict()

    assert result["video"][0]["frame_count"] == 3
    assert output.exists()


def test_distinct_remote_filesystems_run_in_a_real_daft_dataframe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera.mcap"
    _write_compressed_image_mcap(source)
    input_filesystem = _MemoryRemoteFilesystem(
        "tos", {"input/camera.mcap": source.read_bytes()}
    )
    output_filesystem = _MemoryRemoteFilesystem("oss")

    df = daft.from_pydict(
        {
            "mcap_uri": ["tos://input/camera.mcap"],
            "mp4_uri": ["oss://output/camera.mp4"],
        }
    )
    result = df.with_column(
        "video",
        operators.McapVideoExtractor(
            input_filesystem=input_filesystem,
            output_filesystem=output_filesystem,
        ).extract(df["mcap_uri"], df["mp4_uri"]),
    ).to_pydict()["video"][0]

    assert result["output_uri"] == "oss://output/camera.mp4"
    assert result["frame_count"] == 3
    assert result["codec"] == "h264"


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


def test_extract_rejects_unsupported_compressed_images_without_publishing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "png.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(source, format_="png")

    with pytest.raises(ValueError, match="format 'png'.*expected 'h264' or 'jpeg'"):
        operators.McapVideoExtractor().extract(str(source), str(output))

    assert not output.exists()


def test_extract_rejects_unsupported_format_within_selected_stream(
    tmp_path: Path,
) -> None:
    source = tmp_path / "changing-formats.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(
        source,
        packets=_jpeg_packets(),
        frame_formats=("jpeg", "png", "jpeg"),
    )

    with pytest.raises(ValueError, match="format 'png'.*expected 'h264' or 'jpeg'"):
        operators.McapVideoExtractor(h264_preset="ultrafast").extract(
            str(source), str(output)
        )

    assert not output.exists()


def test_extract_rejects_supported_format_change_within_selected_stream(
    tmp_path: Path,
) -> None:
    source = tmp_path / "changing-codecs.mcap"
    output = tmp_path / "camera.mp4"
    h264_packets = _h264_packets()
    jpeg_packets = _jpeg_packets()
    _write_compressed_image_mcap(
        source,
        packets=[h264_packets[0], jpeg_packets[1], h264_packets[2]],
        frame_formats=("h264", "jpeg", "h264"),
    )

    with pytest.raises(ValueError, match="changes compressed video format"):
        operators.McapVideoExtractor().extract(str(source), str(output))

    assert not output.exists()


def test_extract_requires_topic_when_mcap_has_multiple_jpeg_streams(
    tmp_path: Path,
) -> None:
    source = tmp_path / "multi-camera-jpeg.mcap"
    output = tmp_path / "camera.mp4"
    _write_compressed_image_mcap(
        source,
        format_="jpeg",
        packets=_jpeg_packets(),
        topics=("/left/image/compressed", "/right/image/compressed"),
    )

    with pytest.raises(ValueError, match="multiple.*pass topic explicitly"):
        operators.McapVideoExtractor().extract(str(source), str(output))

    assert not output.exists()
    result = operators.McapVideoExtractor(h264_preset="ultrafast").extract(
        str(source),
        str(output),
        topic="/right/image/compressed",
    )
    assert result["topic"] == "/right/image/compressed"
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
