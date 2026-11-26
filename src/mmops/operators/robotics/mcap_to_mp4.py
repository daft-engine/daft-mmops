"""MCAP-to-MP4 extraction for robotics datasets."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO, Iterator, Literal
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

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
_HASH_READ_BYTES = 8 * 1024 * 1024
_SOURCE_READ_AHEAD_BYTES = 1024 * 1024
_JPEG_START_OF_FRAME_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
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

    MCAP input may be a local path, an HTTP(S) URL, or an address supported by
    an injected fsspec filesystem. MP4 output may be a local path or an address
    supported by an injected fsspec filesystem. Remote inputs are consumed in
    MCAP file order without sorting all messages in memory. Local paths use
    fsspec's local filesystem by default. Supplied filesystems must be
    pickleable for Daft worker execution.
    """

    def __init__(
        self,
        *,
        input_filesystem: Any | None = None,
        output_filesystem: Any | None = None,
        temp_dir: str | None = None,
        output_codec: Literal["source", "h264"] = "h264",
        h264_crf: int = 23,
        h264_preset: str = "medium",
    ) -> None:
        """Configure encoding and optionally inject separate fsspec filesystems.

        The input filesystem must support the source URI protocol; HTTP(S)
        input uses urllib when none is supplied. The output filesystem must
        support the destination URI protocol. Local paths default to fsspec's
        local filesystem on each side. Non-local paths require an explicitly
        supplied filesystem. Local output requires hard-link support; remote
        output additionally requires safe no-overwrite publication support.
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
        self.input_filesystem = input_filesystem
        self.output_filesystem = output_filesystem

    @row(return_dtype=MCAP_VIDEO_RESULT_DTYPE)
    def extract(
        self,
        mcap_uri: str | None,
        mp4_uri: str | None,
        topic: str | None = None,
    ) -> dict[str, object] | None:
        """Extract one compressed camera topic into an MP4 object.

        Args:
            mcap_uri: Local, HTTP(S), or fsspec-compatible MCAP input location.
            mp4_uri: Local or fsspec-compatible MP4 output location.
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

        source_protocol = _validate_uri(mcap_uri, "mcap_uri")
        destination_protocol = _validate_uri(mp4_uri, "mp4_uri")
        source_filesystem = self.input_filesystem
        destination_filesystem = self.output_filesystem

        if source_filesystem is None and source_protocol not in (
            "file",
            "http",
            "https",
        ):
            raise ValueError(
                f"input_filesystem is required for the {source_protocol} protocol"
            )
        if destination_filesystem is None and destination_protocol != "file":
            raise ValueError(
                f"output_filesystem is required for the {destination_protocol} protocol"
            )

        if (source_filesystem is None and source_protocol == "file") or (
            destination_filesystem is None and destination_protocol == "file"
        ):
            import fsspec

            local_filesystem = fsspec.filesystem("file")
            if source_filesystem is None and source_protocol == "file":
                source_filesystem = local_filesystem
            if destination_filesystem is None and destination_protocol == "file":
                destination_filesystem = local_filesystem

        if source_filesystem is not None:
            _require_filesystem_protocol(
                source_filesystem, source_protocol, "input_filesystem"
            )
        assert destination_filesystem is not None
        _require_filesystem_protocol(
            destination_filesystem, destination_protocol, "output_filesystem"
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
        self._buffer = b""
        self._buffer_offset = 0

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            buffered = self._buffer[self._buffer_offset :]
            self._buffer = b""
            self._buffer_offset = 0
            return buffered + self._source.read(amount)
        if amount == 0:
            return b""

        available = len(self._buffer) - self._buffer_offset
        if amount <= available:
            end = self._buffer_offset + amount
            result = self._buffer[self._buffer_offset : end]
            self._buffer_offset = end
            if self._buffer_offset == len(self._buffer):
                self._buffer = b""
                self._buffer_offset = 0
            return result

        parts: list[bytes] = []
        if available:
            parts.append(self._buffer[self._buffer_offset :])
            amount -= available
            self._buffer = b""
            self._buffer_offset = 0

        while amount > 0:
            chunk = self._source.read(max(amount, _SOURCE_READ_AHEAD_BYTES))
            if not chunk:
                break
            taken = min(amount, len(chunk))
            parts.append(chunk[:taken])
            amount -= taken
            if taken < len(chunk):
                self._buffer = chunk
                self._buffer_offset = taken
        return b"".join(parts)

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        """Close the underlying HTTP response or fsspec file."""

        _close_stream(self._source)


class _SeekingReadAheadStream:
    """Expose bounded seek/read-ahead over a seekable remote stream."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self._position = int(source.tell())
        self._buffer = b""
        self._buffer_start = self._position

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            buffered = self._buffered_from_position()
            buffer_end = self._buffer_start + len(self._buffer)
            self._source.seek(buffer_end if buffered else self._position, os.SEEK_SET)
            remainder = self._source.read(amount)
            result = buffered + remainder
            self._position += len(result)
            self._discard_buffer()
            return result
        if amount == 0:
            return b""

        parts: list[bytes] = []
        while amount > 0:
            buffered = self._buffered_from_position()
            if buffered:
                taken = min(amount, len(buffered))
                parts.append(buffered[:taken])
                self._position += taken
                amount -= taken
                continue

            self._source.seek(self._position, os.SEEK_SET)
            chunk = self._source.read(max(amount, _SOURCE_READ_AHEAD_BYTES))
            if not chunk:
                break
            self._buffer = chunk
            self._buffer_start = self._position
        return b"".join(parts)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            self._source.seek(offset, os.SEEK_END)
            position = int(self._source.tell())
            self._position = position
            self._discard_buffer()
            return position
        else:
            raise ValueError(f"invalid whence value: {whence}")
        if position < 0:
            raise ValueError("negative seek position")

        buffer_end = self._buffer_start + len(self._buffer)
        if not self._buffer_start <= position <= buffer_end:
            self._source.seek(position, os.SEEK_SET)
            self._discard_buffer(position)
        self._position = position
        return position

    def tell(self) -> int:
        return self._position

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        _close_stream(self._source)

    def _buffered_from_position(self) -> bytes:
        offset = self._position - self._buffer_start
        if 0 <= offset < len(self._buffer):
            return self._buffer[offset:]
        return b""

    def _discard_buffer(self, position: int | None = None) -> None:
        self._buffer = b""
        self._buffer_start = self._position if position is None else position


def _close_stream(source: Any) -> None:
    current = source
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


def _remote_stream(source: Any, *, prefer_seekable: bool) -> Any:
    if prefer_seekable:
        seekable = getattr(source, "seekable", None)
        try:
            if callable(seekable) and seekable():
                position = source.tell()
                source.seek(position, os.SEEK_SET)
                return _SeekingReadAheadStream(source)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return _NonSeekingStream(source)


def _is_local_uri(uri: str) -> bool:
    return urlsplit(uri).scheme.lower() in ("", "file")


def _is_http_uri(uri: str) -> bool:
    return urlsplit(uri).scheme.lower() in ("http", "https")


def _require_filesystem_protocol(filesystem: Any, protocol: str, name: str) -> None:
    supported = getattr(filesystem, "protocol", ())
    if isinstance(supported, str):
        supported = (supported,)
    if protocol not in supported:
        raise ValueError(f"{name} must support the {protocol} protocol")


def _validate_uri(uri: str, name: str) -> str:
    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower() or "file"
    if scheme == "file":
        if parsed.scheme and (
            parsed.netloc not in ("", "localhost")
            or not parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"{name} is not a valid local file URI")
        return scheme
    if not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"{name} is not a valid {scheme} URI")
    if scheme not in ("http", "https") and (parsed.query or parsed.fragment):
        raise ValueError(f"{name} cannot contain a query or fragment")
    return scheme


def _filesystem_path(uri: str) -> str:
    parsed = urlsplit(uri)
    if not parsed.scheme:
        return uri
    if parsed.scheme.lower() == "file":
        return unquote(parsed.path)
    if _is_http_uri(uri):
        return uri
    return f"{parsed.netloc}{parsed.path}"


@contextmanager
def _open_source(
    uri: str,
    source_filesystem: Any | None,
    *,
    prefer_seekable: bool,
) -> Iterator[BinaryIO]:
    if not _is_local_uri(uri) and not _is_http_uri(uri):
        if source_filesystem is None:
            raise RuntimeError("input filesystem is unavailable")
        stream = _remote_stream(
            source_filesystem.open(_filesystem_path(uri), "rb", cache_type="none"),
            prefer_seekable=prefer_seekable,
        )
        try:
            yield stream  # type: ignore[misc]
        finally:
            stream.close()
        return

    if _is_http_uri(uri):
        response = (
            urlopen(uri)
            if source_filesystem is None
            else source_filesystem.open(uri, "rb", cache_type="none")
        )
        stream = _remote_stream(response, prefer_seekable=prefer_seekable)
        try:
            yield stream  # type: ignore[misc]
        finally:
            stream.close()
        return

    if source_filesystem is None:
        raise RuntimeError("local filesystem is unavailable")
    with source_filesystem.open(_filesystem_path(uri), "rb") as input_stream:
        yield input_stream


def _temporary_output(
    destination_uri: str,
    temp_dir: str | None,
    destination_filesystem: Any,
) -> Path:
    if not _is_local_uri(destination_uri):
        directory = Path(temp_dir) if temp_dir is not None else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix="mmops-mcap-",
            # Object-store backends may infer Content-Type from the local name.
            suffix=".mp4",
            dir=directory,
        )
    else:
        destination = Path(_filesystem_path(destination_uri))
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
    unsupported_formats: list[tuple[str, str]],
) -> Iterator[_CompressedFrame]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    decoder_factory = DecoderFactory()
    topics = [requested_topic] if requested_topic is not None else None
    for schema, channel, message in make_reader(input_stream).iter_messages(
        topics=topics, log_time_order=False
    ):
        if schema is None:
            continue

        format_name: str
        timestamp_ns: int
        data: bytes
        if schema.name == "sensor_msgs/msg/CompressedImage":
            decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
            if decoder is None:
                continue
            decoded = decoder(message.data)
            format_name = str(decoded.format)
            timestamp_ns = int(decoded.header.stamp.sec) * 1_000_000_000 + int(
                decoded.header.stamp.nanosec
            )
            data = bytes(decoded.data)
        elif (
            schema.name == "foxglove.CompressedImage"
            and schema.encoding == "jsonschema"
            and channel.message_encoding == "cdr"
        ):
            timestamp_ns, format_name, data = _decode_foxglove_compressed_image(
                message.data
            )
        else:
            continue

        normalized_format = format_name.strip().lower()
        if normalized_format == "h264":
            source_codec: Literal["h264", "mjpeg"] = "h264"
        elif normalized_format == "jpeg":
            source_codec = "mjpeg"
        else:
            if not any(topic == channel.topic for topic, _ in unsupported_formats):
                unsupported_formats.append((channel.topic, format_name))
            continue
        yield _CompressedFrame(
            topic=channel.topic,
            codec=source_codec,
            timestamp_ns=timestamp_ns,
            data=data,
        )


def _decode_single_jpeg(decoder: Any, payload: bytes) -> Any:
    import av

    frames = decoder.decode(av.Packet(payload))
    if len(frames) != 1:
        raise ValueError("each CompressedImage JPEG payload must decode to one frame")
    return frames[0]


def _jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    data = memoryview(payload)
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("JPEG frame is missing a valid start marker")

    offset = 2
    while offset < len(data):
        if data[offset] != 0xFF:
            raise ValueError("JPEG frame contains malformed marker data")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1

        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            continue
        if marker in (0x00, 0xD9, 0xDA) or offset + 2 > len(data):
            break

        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        segment_end = offset + segment_length
        if segment_length < 2 or segment_end > len(data):
            raise ValueError("JPEG frame contains a truncated marker segment")
        if marker in _JPEG_START_OF_FRAME_MARKERS:
            if segment_length < 8:
                raise ValueError("JPEG frame contains a malformed size segment")
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if width <= 0 or height <= 0:
                raise ValueError("JPEG frame has invalid dimensions")
            return width, height
        offset = segment_end

    raise ValueError("JPEG frame is missing a size segment")


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
    first_jpeg_frame: Any | None = None
    unsupported_formats: list[tuple[str, str]] = []
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

            decoded_frame: Any | None = None
            if frame.codec == "mjpeg":
                assert jpeg_decoder is not None
                if frame_count == 0:
                    assert first_jpeg_frame is not None
                    decoded_frame = first_jpeg_frame
                elif output_codec == "h264":
                    decoded_frame = _decode_single_jpeg(jpeg_decoder, frame.data)
                else:
                    width, height = _jpeg_dimensions(frame.data)
                    if width != output_stream.width or height != output_stream.height:
                        raise ValueError(
                            f"topic {selected_topic!r} changes JPEG frame dimensions"
                        )
                if decoded_frame is not None and (
                    decoded_frame.width != output_stream.width
                    or decoded_frame.height != output_stream.height
                ):
                    raise ValueError(
                        f"topic {selected_topic!r} changes JPEG frame dimensions"
                    )

            encoded_packets: list[Any]
            if frame.codec == "mjpeg" and output_codec == "h264":
                assert decoded_frame is not None
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

    selected_unsupported_format = next(
        (
            (topic, format_name)
            for topic, format_name in unsupported_formats
            if topic == selected_topic
        ),
        None,
    )
    if selected_unsupported_format is not None:
        unsupported_topic, unsupported_format = selected_unsupported_format
        raise ValueError(
            f"topic {unsupported_topic!r} uses unsupported CompressedImage "
            f"format {unsupported_format!r}; expected 'h264' or 'jpeg'"
        )

    if (
        frame_count == 0
        or selected_topic is None
        or source_codec is None
        or actual_output_codec is None
    ):
        if unsupported_formats:
            unsupported_topic, unsupported_format = unsupported_formats[0]
            raise ValueError(
                f"topic {unsupported_topic!r} uses unsupported CompressedImage "
                f"format {unsupported_format!r}; expected 'h264' or 'jpeg'"
            )
        topic_description = requested_topic or "a supported CompressedImage topic"
        raise ValueError(f"MCAP does not contain {topic_description}")

    return (
        selected_topic,
        frame_count,
        actual_output_codec,
        source_codec == "h264",
    )


def _sha256_stream(stream: BinaryIO, *, expected_size: int | None = None) -> str:
    digest = hashlib.sha256()
    streamed_size = 0
    maximum_size = None if expected_size is None else expected_size + 1
    while maximum_size is None or streamed_size < maximum_size:
        read_size = _HASH_READ_BYTES
        if maximum_size is not None:
            read_size = min(read_size, maximum_size - streamed_size)
        chunk = stream.read(read_size)
        if not chunk:
            break
        digest.update(chunk)
        streamed_size += len(chunk)
    if expected_size is not None and streamed_size != expected_size:
        raise ValueError(
            f"published object streamed size {streamed_size} does not "
            f"match verified MP4 size {expected_size}"
        )
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

    try:
        with av.open(path) as container:
            if len(container.streams.video) != 1:
                raise ValueError("remuxed MP4 must contain exactly one video stream")
            stream = container.streams.video[0]
            if stream.codec_context.name != expected_codec:
                raise ValueError(
                    f"MP4 uses codec {stream.codec_context.name!r}, not "
                    f"{expected_codec!r}"
                )
            stream.thread_type = "AUTO"
            if stream.duration is None or stream.duration <= 0:
                raise ValueError("remuxed MP4 is missing a positive video duration")
            duration_ns = int(stream.duration * stream.time_base * 1_000_000_000)

            packet_count = 0
            decoded_frame_count = 0
            for packet in container.demux(stream):
                if packet.size > 0:
                    packet_count += 1
                for frame in packet.decode():
                    if reject_b_frames and int(frame.pict_type) == int(
                        av.video.frame.PictureType.B
                    ):
                        raise ValueError(
                            "H.264 streams with B-frame reordering are not supported"
                        )
                    decoded_frame_count += 1
    except av.error.FFmpegError as error:
        raise ValueError("remuxed MP4 contains undecodable video data") from error
    if packet_count != expected_packets:
        raise ValueError(
            f"remuxed MP4 packet count {packet_count} does not match MCAP frame "
            f"count {expected_packets}"
        )
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
    size_bytes: int,
) -> str:
    if not _is_local_uri(destination_uri):
        destination_path = _filesystem_path(destination_uri)
        publish = destination_filesystem.publish_verified_file
        receipt = publish(
            str(temporary),
            destination_path,
            size_bytes=size_bytes,
        )
        destination_filesystem.invalidate_cache(destination_path)
        try:
            sha256, final_verification = _validate_publish_receipt(receipt, size_bytes)
            if final_verification == "readback_required":
                _verify_published_output(
                    destination_filesystem, destination_path, size_bytes, sha256
                )
        except Exception as error:
            raise RuntimeError(
                "remote output verification failed; the final object was "
                "preserved because concurrent writes cannot be ruled out"
            ) from error
        return sha256

    destination = Path(_filesystem_path(destination_uri))
    sha256 = _sha256(temporary)
    try:
        destination_filesystem.link(str(temporary), str(destination))
    except FileExistsError:
        raise FileExistsError(f"output already exists: {destination}") from None
    return sha256


def _validate_publish_receipt(
    receipt: Any, expected_size: int
) -> tuple[str, Literal["sha256", "readback_required"]]:
    if not isinstance(receipt, Mapping):
        raise TypeError("remote publisher must return a publication receipt")
    size_bytes = receipt.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes != expected_size
    ):
        raise ValueError("publication receipt size does not match verified MP4 size")
    sha256 = receipt.get("sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in sha256)
    ):
        raise ValueError("publication receipt sha256 is not a 64-character hex digest")
    final_verification = receipt.get("final_verification")
    if final_verification not in ("sha256", "readback_required"):
        raise ValueError(
            "publication receipt final_verification must be 'sha256' or "
            "'readback_required'"
        )
    return sha256.lower(), final_verification


def _verify_published_output(
    filesystem: Any, path: str, size_bytes: int, sha256: str
) -> None:
    published_size = filesystem.info(path)["size"]
    if published_size != size_bytes:
        raise ValueError(
            f"published object size {published_size} does not "
            f"match verified MP4 size {size_bytes}"
        )
    with filesystem.open(path, "rb", cache_type="none") as stream:
        published_sha256 = _sha256_stream(stream, expected_size=size_bytes)
    if published_sha256 != sha256:
        raise ValueError("published object sha256 does not match verified MP4")


def _preflight_output(destination_uri: str, destination_filesystem: Any) -> None:
    if not _is_local_uri(destination_uri):
        required_methods = (
            "exists",
            "info",
            "open",
            "invalidate_cache",
            "preflight_publish",
            "publish_verified_file",
        )
        if any(
            not callable(getattr(destination_filesystem, name, None))
            for name in required_methods
        ):
            raise RuntimeError(
                "remote filesystem lacks safe no-overwrite publication support"
            )
        destination_path = _filesystem_path(destination_uri)
        destination_filesystem.preflight_publish(destination_path)
        if destination_filesystem.exists(destination_path):
            raise FileExistsError("output already exists at the remote destination")
        return
    if not callable(getattr(destination_filesystem, "link", None)):
        raise RuntimeError("local filesystem lacks hard-link publication support")


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
    _preflight_output(destination_uri, destination_filesystem)
    temporary = _temporary_output(destination_uri, temp_dir, destination_filesystem)
    try:
        with _open_source(
            source_uri,
            source_filesystem,
            prefer_seekable=requested_topic is not None,
        ) as input_stream:
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
        digest = _publish_output(
            temporary,
            destination_uri,
            destination_filesystem,
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
