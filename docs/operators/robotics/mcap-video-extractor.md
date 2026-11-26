# Robotics: McapVideoExtractor usage

`McapVideoExtractor` is a Daft row UDF that turns one camera stream in an MCAP
file into one MP4 file. Each input row produces one metadata result; the MP4
bytes are written to the destination URI, not returned in the DataFrame.

## Install and choose locations

Install the robotics extra in the environment used by every Daft worker:

```bash
python -m pip install 'daft-mmops[robotics]'
```

The extra does not install an object-store backend. Install the backend for
each remote URI scheme in every Daft worker environment and inject compatible
filesystem objects when constructing the extractor.

| Argument | Supported locations |
| --- | --- |
| `mcap_uri` | Local path, `file://` URI, HTTP(S) URL, or an injected filesystem's `scheme://bucket/key` URI |
| `mp4_uri` | Local path, `file://` URI, or an injected filesystem's `scheme://bucket/key` URI |

Local paths and injected remote addresses are accessed through fsspec. For
example, separate injected filesystems may serve `tos://`, `oss://`, or
`s3://` addresses, provided that each object reports the matching `protocol`.
HTTP(S) input uses urllib by default or an injected HTTP(S) filesystem.
Local paths refer to the filesystem visible to the executing worker, which
may differ from the machine creating the DataFrame.

You can pass separate, pickleable fsspec filesystems as `input_filesystem`
and `output_filesystem`. Each supplied object's `protocol` must match its URI
(`file` for local paths, or the URI scheme for a remote filesystem). A remote
input filesystem needs `open`; the operator passes a scheme-less path such as
`bucket/key`. Local paths default to fsspec's local filesystem on each side.
Local output needs a real worker-local path and a `link` method for
no-overwrite hard-link publication. For a local-only job, no filesystem
argument is needed:

```python
from mmops.operators.robotics import McapVideoExtractor

extractor = McapVideoExtractor()
result = extractor.extract("/data/camera.mcap", "/data/camera.mp4")
```

If your application already has input and output filesystems, pass them
separately. They may be the same object when both sides use the same backend
and credentials, or distinct objects when they do not:

```python
from mmops.operators.robotics import McapVideoExtractor


def add_video_column(df, input_filesystem, output_filesystem):
    extractor = McapVideoExtractor(
        input_filesystem=input_filesystem,
        output_filesystem=output_filesystem,
    )
    return df.with_column(
        "video",
        extractor.extract(df["mcap_uri"], df["mp4_uri"]),
    )
```

Call `add_video_column(df, source_fs, destination_fs)` with your application's
existing objects, then execute the returned DataFrame with `to_pydict()`.
Daft serializes the operator for worker execution, including both supplied
filesystems. Check that neither pickle contains access keys, tokens, or
environment-specific endpoints. A filesystem built on the submitting process
with static credentials typically serializes them and is **not** suitable for
this injection mode. Use pickle-safe worker-side identity or credential
providers.

### Remote output publication contract

A standard fsspec filesystem is sufficient for remote **input**, but fsspec
does not guarantee atomic create-only publication for every backend. Remote
**output** therefore requires a compatible filesystem or wrapper with these
additional methods:

```python
def preflight_publish(path: str) -> None: ...
def publish_verified_file(
    local_path: str, path: str, *, size_bytes: int
) -> dict[str, object]:
    return {
        "size_bytes": size_bytes,
        "sha256": "<64 lowercase hex characters>",
        "final_verification": "sha256",  # or "readback_required"
    }
```

`preflight_publish` must reject storage settings that defeat create-only
publication. `publish_verified_file` must stage and verify the complete file,
then conditionally create the final key without overwriting a concurrent
writer and safely clean up its own staging object. It must arrange final-key
verification through one of the two receipt modes below. While reading
`local_path` for upload, it must calculate the SHA-256 and return that digest
and the observed size in the receipt. This avoids a separate local-file hash
pass.

Set `final_verification` to `"sha256"` only when a cryptographic SHA-256
reported by the backend and bound to the exact final object or version matches
the receipt. An ETag or caller-written metadata is not sufficient. In that
case the operator trusts the wrapper's attestation and does not download the
final object. If no such backend attestation exists, return
`"readback_required"`; the operator then checks `info`, downloads the final
object once, and verifies its exact size and SHA-256. That read is bounded to
the expected size plus one byte so growth between `info` and `open` fails
closed.

The publisher must raise `FileExistsError` on a destination conflict and must
not delete a final object whose identity is uncertain. The operator validates
every receipt and likewise preserves an ambiguously published final object on
verification failure.
The output filesystem must also provide `exists` and `invalidate_cache`.
An unwrapped fsspec backend without this contract fails closed before the
input is read; installing its package alone does not enable remote output.

## Configure storage credentials

The application that creates each filesystem is responsible for its endpoint,
region, access key, secret key, or temporary-token configuration. This UDF
does **not** read backend-specific environment variables, register object
store implementations, or construct a remote filesystem. Setting environment
variables without passing the corresponding `input_filesystem=` or
`output_filesystem=` does not enable a remote scheme: the UDF fails before
reading the input. If a filesystem or publication wrapper uses worker
environment variables or a secret provider, configure that provider in every
worker where the UDF can execute.
Do not commit credentials, put them in DataFrame columns or serialized
filesystem objects, or print them in logs.

## Run the UDF

```python
import daft

from mmops.operators.robotics import McapVideoExtractor

df = daft.from_pydict(
    {
        "mcap_uri": ["s3://input-bucket/captures/camera.mcap"],
        "mp4_uri": ["tos://output-bucket/run-001/camera.mp4"],
    }
)

def run(df, input_filesystem=None, output_filesystem=None):
    extractor = McapVideoExtractor(
        input_filesystem=input_filesystem,
        output_filesystem=output_filesystem,
        output_codec="h264",
        temp_dir="/var/tmp/mmops",
    )
    result = df.with_column(
        "video",
        extractor.extract(df["mcap_uri"], df["mp4_uri"]),
    )
    return result.select("video").to_pydict()  # Executes the Daft plan.
```

Call `run(df, source_fs, destination_fs)` after creating the two matching
filesystems in your application; `destination_fs` must implement the remote
publication contract above. For a local-only job, use local paths and call
`run(df)`.
Use unique output URIs for each run. If the MCAP contains multiple supported
camera topics, select one with the `topic` argument, for example
pass `topic="/camera/front"` to `extractor.extract(...)` inside `run`.
The UDF also accepts local source and destination paths in the same call.

`output_codec="h264"` (the default) remuxes source H.264 without re-encoding
and decodes JPEG frames before CPU encoding with `libx264`.
`output_codec="source"` remuxes H.264 or JPEG as H.264 or MJPEG, respectively.
MJPEG source remuxing decodes the first frame to establish stream parameters,
checks later frame dimensions from their JPEG headers, and performs one full
decode of the completed MP4 during validation.
For JPEG-to-H.264 encoding, `h264_crf` is an integer from `0` through `51`
(default `23`), and `h264_preset` is an x264 preset (default `"medium"`). These
are constructor options shared by all rows processed by that extractor.

Supported camera schemas are ROS2 `sensor_msgs/msg/CompressedImage` and
CDR-encoded `foxglove.CompressedImage`. For both schemas, the message's
`format` field is the sole source of the compressed video format: `h264`
selects H.264 remuxing and `jpeg` selects the JPEG processing path. The schema
name is used only to deserialize the message. Camera messages are read in MCAP
**file order**, without a whole-file log-time sort;
their header timestamps must be strictly increasing in that order. JPEG frames
must keep one resolution throughout the selected topic; the UDF rejects a
dimension change instead of silently scaling frames. Source H.264 with B-frame
reordering is rejected. The generated MP4 is decoded for validation before
publication.

## Read the result

The `video` column is a struct with these fields:

| Field | Meaning |
| --- | --- |
| `output_uri` | Destination URI that received the MP4 |
| `topic` | Selected camera topic |
| `frame_count` | Number of video frames |
| `duration_ns` | Final MP4 timeline duration in nanoseconds, including the inferred last-frame duration |
| `size_bytes` | Published MP4 size |
| `sha256` | Hex SHA-256 digest of the published MP4 |
| `codec` | Actual MP4 video codec, `h264` or `mjpeg` |

Paired null `mcap_uri` and `mp4_uri` values produce a null result. A partial
null pair, unsupported URI or camera format, missing or ambiguous topic,
invalid timestamps, undecodable video, and an existing output are errors.

## Plan storage and retries

Each row needs worker-local temporary disk roughly equal to its output MP4
size. Remote output uses `temp_dir` when set, otherwise the system temporary
directory; local output uses the destination directory. Make the selected
directory writable on every worker. Input messages are not globally sorted in
memory. Remote input uses up to 1 MiB of read-ahead to coalesce MCAP's small
reads. When `topic` is explicit and the injected remote stream supports seek,
the reader uses the MCAP summary and chunk indexes to skip nonmatching chunks
and unrelated attachment bodies. A missing index, a non-seekable stream, or
automatic topic selection falls back to one sequential pass. Peak memory still
depends on the largest MCAP chunk or record, codec buffers, the 8 MiB digest
read buffer, and worker concurrency. An unusually large attachment can
increase sequential-read cost. JPEG-to-H.264 conversion and final validation
use CPU. Final validation enables FFmpeg's automatic frame and slice decode
threading; account for that CPU use when choosing Daft worker and operator
concurrency. This UDF does not use GPU/NVENC.

The operator writes and verifies a temporary MP4 before publishing. Its
remote publication wrapper is responsible for staging, conditional final
creation, and safe staging cleanup. The operator checks whether the final
path already exists before decoding, but that check is only an early failure:
the wrapper must enforce create-only publication again at commit time to
cover races. If final verification is ambiguous, the operator preserves the
final key rather than risk deleting a concurrent writer's object; inspect
that key before retrying. Backend-specific conditions such as bucket
versioning belong in the wrapper's preflight check.

Daft may retry a row, but cannot make these external writes exactly once.
Use unique run-scoped destination URIs and handle an already-existing output
as an explicit conflict. Do not rely on global row order for execution.
