"""MCAP-to-MP4 extraction for robotics datasets."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO, Iterator, Literal
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4
import warnings

import pyarrow as pa

from ..._decorators import operator, row


MCAP_VIDEO_RESULT_DTYPE = pa.struct(
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

_H264_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }
)


@dataclass(frozen=True)
class _CompressedFrame:
    topic: str
    codec: Literal["h264", "mjpeg"]
    timestamp_ns: int
    data: bytes


@operator
class McapVideoExtractor:
    """Extract compressed camera messages from MCAP into an MP4 object.

    MCAP input may be a local path, a ``tos://`` object URI, or an
    ``http://``/``https://`` URL. MP4 output may be a local path or a
    ``tos://`` object URI. HTTP inputs are consumed as non-seeking streams and
    are not downloaded into memory as a whole. Local paths use fsspec's local
    filesystem. A compatible filesystem may be passed to the constructor;
    it must be pickleable for Daft worker execution.
    """

    def __init__(
        self,
        *,
        filesystem: Any | None = None,
        temp_dir: str | None = None,
        output_codec: Literal["source", "h264"] = "h264",
        h264_crf: int = 23,
        h264_preset: str = "medium",
    ) -> None:
        """Configure output encoding and optionally inject a fsspec filesystem.

        A supplied filesystem is used for matching local or ``tos://`` URIs.
        For mixed TOS/local jobs, local paths use fsspec's local filesystem
        unless the supplied filesystem supports the ``file`` protocol too.
        Without an injected TOS filesystem, worker environment variables are
        used to create one lazily.
        """
        if output_codec not in ("source", "h264"):
            raise ValueError(
                "output_codec must be 'source' or 'h264', " f"got {output_codec!r}"
            )
        if (
            isinstance(h264_crf, bool)
            or not isinstance(h264_crf, int)
            or not 0 <= h264_crf <= 51
        ):
            raise ValueError(
                f"h264_crf must be an integer from 0 through 51, got {h264_crf!r}"
            )
        if h264_preset not in _H264_PRESETS:
            choices = ", ".join(sorted(_H264_PRESETS))
            raise ValueError(
                f"h264_preset must be one of {choices}, got {h264_preset!r}"
            )
        self.temp_dir = temp_dir
        self.output_codec = output_codec
        self.h264_crf = h264_crf
        self.h264_preset = h264_preset
        self._provided_filesystem = filesystem
        self._tos_filesystem_cache: Any | None = None
        self._local_filesystem_cache: Any | None = None

    def _filesystem_for(
        self,
        protocol: Literal["file", "tos"],
        *,
        allow_default_local: bool = False,
    ) -> Any:
        provided = self._provided_filesystem
        if provided is not None:
            supported = getattr(provided, "protocol", ())
            if isinstance(supported, str):
                supported = (supported,)
            if protocol in supported:
                return provided
            if protocol == "tos" or not allow_default_local:
                raise ValueError(f"filesystem must support the {protocol} protocol")
        if protocol == "tos":
            return self._tos_filesystem()
        if self._local_filesystem_cache is None:
            import fsspec

            self._local_filesystem_cache = fsspec.filesystem("file")
        return self._local_filesystem_cache

    def _tos_filesystem(self) -> Any:
        endpoint = os.environ.get("TOS_ENDPOINT")
        region = os.environ.get("TOS_REGION")
        if not endpoint or not region:
            raise ValueError(
                "TOS_ENDPOINT and TOS_REGION are required in the worker environment "
                "when an input or output uses tos://"
            )
        key = os.environ.get("TOS_ACCESS_KEY")
        secret = os.environ.get("TOS_SECRET_KEY")
        if not key or not secret:
            raise ValueError(
                "TOS_ACCESS_KEY and TOS_SECRET_KEY are required in the worker "
                "environment when an input or output uses tos://"
            )
        if self._tos_filesystem_cache is None:
            import fsspec

            fsspec.register_implementation("tos", "tosfs.TosFileSystem")
            self._tos_filesystem_cache = fsspec.filesystem(
                "tos",
                endpoint=endpoint,
                region=region,
                key=key,
                secret=secret,
                session_token=os.environ.get("TOS_SECURITY_TOKEN"),
            )
        return self._tos_filesystem_cache

    @row(return_dtype=MCAP_VIDEO_RESULT_DTYPE)
    def extract(
        self,
        mcap_uri: str | None,
        mp4_uri: str | None,
        topic: str | None = None,
    ) -> dict[str, object] | None:
        """Extract one compressed camera topic into an MP4 object.

        Args:
            mcap_uri: Local, TOS, HTTP, or HTTPS MCAP input location.
            mp4_uri: Local or TOS MP4 output location.
            topic: Camera topic to select when the MCAP has multiple supported
                compressed-image topics.

        Returns:
            Output URI, selected topic, frame count, duration, size, digest,
            and actual codec, or ``None`` for a paired null input/output row.

        Raises:
            ValueError: The URI pair, URI scheme, topic, timestamps, codec, or
                generated MP4 violates the public extraction contract.
            FileExistsError: The output location already exists.
        """

        if mcap_uri is None and mp4_uri is None:
            return None
        if mcap_uri is None or mp4_uri is None:
            raise ValueError(
                "mcap_uri and mp4_uri must either both be null or both be set"
            )

        _validate_source_uri(mcap_uri)
        _validate_destination_uri(mp4_uri)
        uses_tos = _is_tos_uri(mcap_uri) or _is_tos_uri(mp4_uri)
        source_filesystem = (
            None
            if _is_http_uri(mcap_uri)
            else self._filesystem_for(
                "tos" if _is_tos_uri(mcap_uri) else "file",
                allow_default_local=uses_tos,
            )
        )
        destination_filesystem = self._filesystem_for(
            "tos" if _is_tos_uri(mp4_uri) else "file",
            allow_default_local=uses_tos,
        )
        return _extract_uri(
            mcap_uri,
            mp4_uri,
            topic,
            source_filesystem=source_filesystem,
            destination_filesystem=destination_filesystem,
            temp_dir=self.temp_dir,
            output_codec=self.output_codec,
            h264_crf=self.h264_crf,
            h264_preset=self.h264_preset,
        )


class _NonSeekingStream:
    def __init__(self, source: Any) -> None:
        self._source = source

    def read(self, amount: int = -1) -> bytes:
        return self._source.read(amount)

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        """Close the underlying HTTP response or fsspec file."""

        current = self._source
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            close = getattr(current, "close", None)
            if callable(close):
                close()
                return
            content = getattr(current, "content", None)
            if content is not None and content is not current:
                current = content
                continue
            data = getattr(current, "data", None)
            if data is not None and data is not current:
                current = data
                continue
            return


def _is_tos_uri(uri: str) -> bool:
    return urlsplit(uri).scheme.lower() == "tos"


def _is_http_uri(uri: str) -> bool:
    return urlsplit(uri).scheme.lower() in ("http", "https")


def _tos_location(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme.lower() != "tos"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
    ):
        raise ValueError(f"invalid TOS object URI: {uri!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"TOS object URI cannot contain query or fragment: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _validate_source_uri(uri: str) -> None:
    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower()
    if not scheme:
        return
    if scheme == "tos":
        _tos_location(uri)
        return
    if scheme in ("http", "https"):
        if not parsed.netloc:
            raise ValueError(f"invalid HTTP input URI: {uri!r}")
        return
    raise ValueError(
        f"mcap_uri uses unsupported URI scheme {scheme!r}; expected a local "
        "path, tos://, http://, or https://"
    )


def _validate_destination_uri(uri: str) -> None:
    scheme = urlsplit(uri).scheme.lower()
    if not scheme:
        return
    if scheme == "tos":
        _tos_location(uri)
        return
    raise ValueError(
        f"mp4_uri uses unsupported URI scheme {scheme!r}; expected a local "
        "path or tos://"
    )


@contextmanager
def _open_source(uri: str, source_filesystem: Any | None) -> Iterator[BinaryIO]:
    if _is_tos_uri(uri):
        if source_filesystem is None:
            raise RuntimeError("TOS filesystem is unavailable")
        bucket, key = _tos_location(uri)
        stream = _NonSeekingStream(
            source_filesystem.open(f"{bucket}/{key}", "rb", cache_type="none")
        )
        try:
            yield stream  # type: ignore[misc]
        finally:
            stream.close()
        return

    if _is_http_uri(uri):
        response = urlopen(uri)
        stream = _NonSeekingStream(response)
        try:
            yield stream  # type: ignore[misc]
        finally:
            stream.close()
        return

    if source_filesystem is None:
        raise RuntimeError("local filesystem is unavailable")
    with source_filesystem.open(uri, "rb") as input_stream:
        yield input_stream


def _temporary_output(
    destination_uri: str,
    temp_dir: str | None,
    destination_filesystem: Any,
) -> Path:
    if _is_tos_uri(destination_uri):
        directory = Path(temp_dir) if temp_dir is not None else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix="mmops-mcap-",
            # tosfs infers Content-Type from the local filename on put_file.
            suffix=".mp4",
            dir=directory,
        )
    else:
        destination = Path(destination_uri)
        if destination_filesystem.exists(str(destination)):
            raise FileExistsError(f"output already exists: {destination}")
        destination_filesystem.makedirs(str(destination.parent), exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    os.close(descriptor)
    return Path(name)


def _read_cdr_string(payload: bytes, offset: int, field_name: str) -> tuple[str, int]:
    if offset + 4 > len(payload):
        raise ValueError(f"Foxglove CompressedImage {field_name} length is truncated")
    length = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    end = offset + length
    if length == 0 or end > len(payload) or payload[end - 1] != 0:
        raise ValueError(f"Foxglove CompressedImage {field_name} is malformed")
    try:
        value = payload[offset : end - 1].decode()
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Foxglove CompressedImage {field_name} is not UTF-8"
        ) from error
    return value, (end + 3) & ~3


def _decode_foxglove_compressed_image(payload: bytes) -> tuple[int, str, bytes]:
    if len(payload) < 12 or payload[:4] != b"\0\0\0\0":
        raise ValueError(
            "Foxglove CompressedImage must use the supported little-endian CDR "
            "layout"
        )
    seconds, nanoseconds = struct.unpack_from("<iI", payload, 4)
    if seconds < 0 or nanoseconds >= 1_000_000_000:
        raise ValueError("Foxglove CompressedImage timestamp is invalid")
    _, offset = _read_cdr_string(payload, 12, "frame_id")
    format_name, offset = _read_cdr_string(payload, offset, "format")
    if offset + 4 > len(payload):
        raise ValueError("Foxglove CompressedImage data length is truncated")
    data_length = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    end = offset + data_length
    padded_end = (end + 3) & ~3
    if padded_end != len(payload) or any(payload[end:padded_end]):
        raise ValueError("Foxglove CompressedImage data is malformed")
    return seconds * 1_000_000_000 + nanoseconds, format_name, payload[offset:end]


def _iter_compressed_frames(
    input_stream: BinaryIO,
    requested_topic: str | None,
    unsupported_formats: list[tuple[str, str, str]],
) -> Iterator[_CompressedFrame]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    decoder_factory = DecoderFactory()
    topics = [requested_topic] if requested_topic is not None else None
    for schema, channel, message in make_reader(input_stream).iter_messages(
        topics=topics
    ):
        if schema is None:
            continue
        if schema.name == "sensor_msgs/msg/CompressedImage":
            decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
            if decoder is None:
                continue
            decoded = decoder(message.data)
            format_name = decoded.format.lower()
            if format_name != "h264":
                unsupported_formats.append((channel.topic, decoded.format, "h264"))
                continue
            timestamp_ns = int(decoded.header.stamp.sec) * 1_000_000_000 + int(
                decoded.header.stamp.nanosec
            )
            yield _CompressedFrame(
                topic=channel.topic,
                codec="h264",
                timestamp_ns=timestamp_ns,
                data=bytes(decoded.data),
            )
            continue

        if (
            schema.name == "foxglove.CompressedImage"
            and schema.encoding == "jsonschema"
            and channel.message_encoding == "cdr"
        ):
            timestamp_ns, format_name, data = _decode_foxglove_compressed_image(
                message.data
            )
            if format_name.lower() != "jpeg":
                unsupported_formats.append((channel.topic, format_name, "jpeg"))
                continue
            yield _CompressedFrame(
                topic=channel.topic,
                codec="mjpeg",
                timestamp_ns=timestamp_ns,
                data=data,
            )


def _decode_single_jpeg(decoder: Any, payload: bytes) -> Any:
    import av

    frames = decoder.decode(av.Packet(payload))
    if len(frames) != 1:
        raise ValueError(
            "each Foxglove CompressedImage JPEG payload must decode to one frame"
        )
    return frames[0]


def _write_video(
    input_stream: BinaryIO,
    temporary: Path,
    requested_topic: str | None,
    *,
    output_codec: Literal["source", "h264"],
    h264_crf: int,
    h264_preset: str,
) -> tuple[str, int, Literal["h264", "mjpeg"], bool]:
    import av

    selected_topic: str | None = None
    source_codec: Literal["h264", "mjpeg"] | None = None
    actual_output_codec: Literal["h264", "mjpeg"] | None = None
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    last_frame_duration_ns: int | None = None
    frame_count = 0
    pending_packet: Any | None = None
    output_stream: Any | None = None
    jpeg_decoder: Any | None = None
    unsupported_formats: list[tuple[str, str, str]] = []
    time_base = Fraction(1, 1_000_000_000)

    with av.open(str(temporary), mode="w", format="mp4") as output_container:
        for frame in _iter_compressed_frames(
            input_stream,
            requested_topic,
            unsupported_formats,
        ):
            if selected_topic is None:
                selected_topic = frame.topic
                source_codec = frame.codec
            elif frame.topic != selected_topic:
                raise ValueError(
                    "MCAP contains multiple supported CompressedImage topics; "
                    "pass topic explicitly"
                )
            elif frame.codec != source_codec:
                raise ValueError(
                    f"topic {selected_topic!r} changes compressed video format"
                )

            if first_timestamp_ns is None:
                first_timestamp_ns = frame.timestamp_ns
            if (
                last_timestamp_ns is not None
                and frame.timestamp_ns <= last_timestamp_ns
            ):
                raise ValueError(
                    f"topic {selected_topic!r} must have strictly increasing "
                    "camera timestamps"
                )
            relative_timestamp_ns = frame.timestamp_ns - first_timestamp_ns

            if output_stream is None:
                if frame.codec == "h264":
                    actual_output_codec = "h264"
                    output_stream = output_container.add_stream("h264")
                    output_stream.time_base = time_base
                else:
                    jpeg_decoder = av.CodecContext.create("mjpeg", "r")
                    first_jpeg_frame = _decode_single_jpeg(jpeg_decoder, frame.data)
                    if output_codec == "source":
                        actual_output_codec = "mjpeg"
                        output_stream = output_container.add_stream("mjpeg")
                        output_stream.width = first_jpeg_frame.width
                        output_stream.height = first_jpeg_frame.height
                        output_stream.pix_fmt = first_jpeg_frame.format.name
                    else:
                        actual_output_codec = "h264"
                        output_stream = output_container.add_stream(
                            "libx264",
                            options={
                                "crf": str(h264_crf),
                                "preset": h264_preset,
                                "x264-params": "bframes=0",
                            },
                        )
                        output_stream.width = first_jpeg_frame.width
                        output_stream.height = first_jpeg_frame.height
                        output_stream.pix_fmt = "yuv420p"
                    output_stream.time_base = time_base
                    output_stream.codec_context.time_base = time_base

            encoded_packets: list[Any]
            if frame.codec == "mjpeg" and output_codec == "h264":
                assert jpeg_decoder is not None
                if frame_count == 0:
                    decoded_frame = first_jpeg_frame
                else:
                    decoded_frame = _decode_single_jpeg(jpeg_decoder, frame.data)
                decoded_frame = decoded_frame.reformat(
                    output_stream.width,
                    output_stream.height,
                    "yuv420p",
                )
                decoded_frame.pts = relative_timestamp_ns
                decoded_frame.time_base = time_base
                encoded_packets = list(output_stream.encode(decoded_frame))
            else:
                packet = av.Packet(frame.data)
                packet.pts = relative_timestamp_ns
                packet.dts = packet.pts
                packet.time_base = time_base
                packet.stream = output_stream
                encoded_packets = [packet]

            for packet in encoded_packets:
                packet.stream = output_stream
                if pending_packet is not None:
                    last_frame_duration_ns = packet.pts - pending_packet.pts
                    pending_packet.duration = last_frame_duration_ns
                    output_container.mux(pending_packet)
                pending_packet = packet

            last_timestamp_ns = frame.timestamp_ns
            frame_count += 1

        if (
            source_codec == "mjpeg"
            and output_codec == "h264"
            and output_stream is not None
        ):
            for packet in output_stream.encode(None):
                packet.stream = output_stream
                if pending_packet is not None:
                    last_frame_duration_ns = packet.pts - pending_packet.pts
                    pending_packet.duration = last_frame_duration_ns
                    output_container.mux(pending_packet)
                pending_packet = packet

        if pending_packet is not None:
            pending_packet.duration = last_frame_duration_ns or 1
            output_container.mux(pending_packet)

    if (
        frame_count == 0
        or selected_topic is None
        or source_codec is None
        or actual_output_codec is None
    ):
        if unsupported_formats:
            unsupported_topic, unsupported_format, expected_format = (
                unsupported_formats[0]
            )
            raise ValueError(
                f"topic {unsupported_topic!r} uses unsupported CompressedImage "
                f"format {unsupported_format!r}; expected {expected_format!r}"
            )
        topic_description = requested_topic or "a supported CompressedImage topic"
        raise ValueError(f"MCAP does not contain {topic_description}")

    return (
        selected_topic,
        frame_count,
        actual_output_codec,
        source_codec == "h264",
    )


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _verify_mp4(
    path: Path,
    expected_packets: int,
    expected_codec: Literal["h264", "mjpeg"],
    *,
    reject_b_frames: bool,
) -> int:
    import av

    with av.open(path) as container:
        if len(container.streams.video) != 1:
            raise ValueError("remuxed MP4 must contain exactly one video stream")
        stream = container.streams.video[0]
        if stream.codec_context.name != expected_codec:
            raise ValueError(
                f"MP4 uses codec {stream.codec_context.name!r}, not "
                f"{expected_codec!r}"
            )
        packet_count = sum(1 for packet in container.demux(stream) if packet.size > 0)
        if stream.duration is None or stream.duration <= 0:
            raise ValueError("remuxed MP4 is missing a positive video duration")
        duration_ns = int(stream.duration * stream.time_base * 1_000_000_000)
    if packet_count != expected_packets:
        raise ValueError(
            f"remuxed MP4 packet count {packet_count} does not match MCAP frame "
            f"count {expected_packets}"
        )

    try:
        with av.open(path) as container:
            decoded_frame_count = 0
            for frame in container.decode(video=0):
                if reject_b_frames and int(frame.pict_type) == int(
                    av.video.frame.PictureType.B
                ):
                    raise ValueError(
                        "H.264 streams with B-frame reordering are not supported"
                    )
                decoded_frame_count += 1
    except av.error.FFmpegError as error:
        raise ValueError("remuxed MP4 contains undecodable H.264 data") from error
    if decoded_frame_count != expected_packets:
        raise ValueError(
            f"remuxed MP4 decoded frame count {decoded_frame_count} does not "
            f"match MCAP frame count {expected_packets}"
        )
    return duration_ns


def _publish_output(
    temporary: Path,
    destination_uri: str,
    destination_filesystem: Any,
    sha256: str,
    size_bytes: int,
) -> None:
    if _is_tos_uri(destination_uri):
        tos_filesystem = destination_filesystem
        # fsspec has no atomic conditional copy or bucket-versioning interface.
        # Keep these publication-only operations on the tosfs backend client.
        tos_client = getattr(tos_filesystem, "tos_client", None)
        if tos_client is None:
            raise RuntimeError("TOS filesystem lacks conditional publication support")
        bucket, key = _tos_location(destination_uri)
        _require_unversioned_tos_bucket(tos_client, bucket)
        staging_key = f"{key}.mmops-staging-{uuid4().hex}"
        staging_path = f"{bucket}/{staging_key}"
        final_path = f"{bucket}/{key}"
        staging_upload_attempted = False
        try:
            staging_upload_attempted = True
            tos_filesystem.put_file(str(temporary), staging_path)
            _verify_tos_object(tos_filesystem, staging_path, size_bytes, sha256)
            staged = tos_client.head_object(bucket, staging_key)
            source_etag = getattr(staged, "etag", None)
            if not source_etag:
                raise ValueError("staged TOS object is missing an ETag")

            copied = tos_client.copy_object(
                bucket,
                key,
                bucket,
                staging_key,
                copy_source_if_match=source_etag,
                forbid_overwrite=True,
            )
            if getattr(copied, "version_id", None) is not None:
                _remove_created_tos_copy(tos_client, bucket, key, copied)
                raise RuntimeError(
                    "TOS bucket versioning changed during publication; the new "
                    "object version was removed and the final key was not accepted"
                )
            tos_filesystem.invalidate_cache(final_path)
            try:
                _verify_tos_object(tos_filesystem, final_path, size_bytes, sha256)
            except Exception as error:
                raise RuntimeError(
                    "TOS final object verification failed; the unversioned final "
                    "key was preserved because it cannot be deleted safely under "
                    "concurrent writes"
                ) from error
        finally:
            if staging_upload_attempted:
                try:
                    _remove_tos_staging_object(
                        tos_filesystem,
                        tos_client,
                        bucket,
                        staging_key,
                    )
                except Exception as error:
                    warnings.warn(
                        f"failed to remove TOS staging object {staging_key!r}: "
                        f"{error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        return

    destination = Path(destination_uri)
    try:
        destination_filesystem.link(str(temporary), str(destination))
    except FileExistsError:
        raise FileExistsError(f"output already exists: {destination}") from None


def _verify_tos_object(
    tos_filesystem: Any, path: str, size_bytes: int, sha256: str
) -> None:
    published_size = tos_filesystem.info(path)["size"]
    if published_size != size_bytes:
        raise ValueError(
            f"published TOS object size {published_size} does not "
            f"match verified MP4 size {size_bytes}"
        )
    with tos_filesystem.open(path, "rb", cache_type="none") as stream:
        published_sha256 = _sha256_stream(stream)
    if published_sha256 != sha256:
        raise ValueError("published TOS object sha256 does not match verified MP4")


def _require_unversioned_tos_bucket(tos_client: Any, bucket: str) -> None:
    versioning = tos_client.get_bucket_version(bucket).status
    status = getattr(versioning, "value", versioning)
    if status not in (None, "", "Unknown"):
        raise ValueError(
            f"TOS bucket {bucket!r} versioning is {status}; no-overwrite "
            "publication requires a bucket where versioning has never been enabled"
        )


def _remove_tos_staging_object(
    tos_filesystem: Any,
    tos_client: Any,
    bucket: str,
    staging_key: str,
) -> None:
    version_id = None
    try:
        staged = tos_client.head_object(bucket, staging_key)
        version_id = getattr(staged, "version_id", None)
    except Exception:
        pass
    if version_id is None:
        tos_filesystem.rm(f"{bucket}/{staging_key}")
    else:
        tos_client.delete_object(bucket, staging_key, version_id=version_id)


def _remove_created_tos_copy(
    tos_client: Any,
    bucket: str,
    key: str,
    copied: Any,
) -> None:
    version_id = getattr(copied, "version_id", None)
    if version_id is not None:
        tos_client.delete_object(bucket, key, version_id=version_id)


def _extract_uri(
    source_uri: str,
    destination_uri: str,
    requested_topic: str | None,
    *,
    source_filesystem: Any | None,
    destination_filesystem: Any,
    temp_dir: str | None,
    output_codec: Literal["source", "h264"],
    h264_crf: int,
    h264_preset: str,
) -> dict[str, object]:
    temporary = _temporary_output(destination_uri, temp_dir, destination_filesystem)
    try:
        with _open_source(source_uri, source_filesystem) as input_stream:
            selected_topic, frame_count, actual_output_codec, source_is_h264 = (
                _write_video(
                    input_stream,
                    temporary,
                    requested_topic,
                    output_codec=output_codec,
                    h264_crf=h264_crf,
                    h264_preset=h264_preset,
                )
            )
        duration_ns = _verify_mp4(
            temporary,
            frame_count,
            actual_output_codec,
            reject_b_frames=source_is_h264,
        )
        size_bytes = temporary.stat().st_size
        digest = _sha256(temporary)
        _publish_output(
            temporary,
            destination_uri,
            destination_filesystem,
            digest,
            size_bytes,
        )
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "output_uri": destination_uri,
        "topic": selected_topic,
        "frame_count": frame_count,
        "duration_ns": duration_ns,
        "size_bytes": size_bytes,
        "sha256": digest,
        "codec": actual_output_codec,
    }
