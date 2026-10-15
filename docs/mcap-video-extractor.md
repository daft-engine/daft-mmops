# McapVideoExtractor usage

`McapVideoExtractor` is a Daft row UDF that turns one camera stream in an MCAP
file into one MP4 file. Each input row produces one metadata result; the MP4
bytes are written to the destination URI, not returned in the DataFrame.

## Install and choose locations

Install the robotics extra in the environment used by every Daft worker:

```bash
python -m pip install 'daft-mmops[robotics]'
```

| Argument | Supported locations |
| --- | --- |
| `mcap_uri` | Local path, `tos://bucket/key`, HTTP URL, or HTTPS URL |
| `mp4_uri` | Local path or `tos://bucket/key` |

The `tos://` object address is accessed through fsspec. This UDF currently
recognizes that object-store scheme only; an installed fsspec implementation
for another scheme does not make `s3://` or other URIs valid here. HTTP(S) is
input-only. Local paths refer to the filesystem visible to the executing
worker, which may differ from the machine creating the DataFrame.

## Configure TOS credentials

Set these environment variables **on every Daft worker before execution**
when either URI uses `tos://`:

| Variable | Purpose | Required |
| --- | --- | --- |
| `TOS_ENDPOINT` | TOS endpoint host, for example `tos-cn-beijing.volces.com` | Yes |
| `TOS_REGION` | Region, for example `cn-beijing` | Yes |
| `TOS_ACCESS_KEY` | Access key (AK) | Yes |
| `TOS_SECRET_KEY` | Secret key (SK) | Yes |
| `TOS_SECURITY_TOKEN` | Session token for temporary credentials | Only when issued |

For a local Daft process, export the values in the shell that launches Python:

```bash
export TOS_ENDPOINT='tos-cn-beijing.volces.com'
export TOS_REGION='cn-beijing'
export TOS_ACCESS_KEY='<access-key>'
export TOS_SECRET_KEY='<secret-key>'
# If using temporary credentials:
export TOS_SECURITY_TOKEN='<session-token>'
python pipeline.py
```

Replace the placeholders using your secret manager or deployment system. Do
not commit credentials, put them in DataFrame columns or UDF constructor
arguments, or print them in logs. For distributed Daft execution, configure
the worker environment too; setting variables only in the submitting process
is not sufficient. Local and HTTP(S)-only jobs do not require TOS variables.
The UDF passes these values to the fsspec filesystem as `endpoint`, `region`,
`key`, `secret`, and `session_token`, respectively.

## Run the UDF

```python
import daft

from mmops.operators.robotics import McapVideoExtractor

df = daft.from_pydict(
    {
        "mcap_uri": ["tos://input-bucket/captures/camera.mcap"],
        "mp4_uri": ["tos://output-bucket/run-001/camera.mp4"],
    }
)

extractor = McapVideoExtractor(
    output_codec="h264",
    temp_dir="/var/tmp/mmops",
)
result = df.with_column(
    "video",
    extractor.extract(df["mcap_uri"], df["mp4_uri"]),
)
print(result.select("video").to_pydict())  # Executes the Daft plan.
```

Use unique output URIs for each run. If the MCAP contains multiple supported
camera topics, select one with the `topic` argument, for example
`extractor.extract(df["mcap_uri"], df["mp4_uri"], topic="/camera/front")`.
The UDF also accepts local source and destination paths in the same call.

`output_codec="h264"` (the default) remuxes source H.264 without re-encoding
and decodes Foxglove JPEG frames before CPU encoding with `libx264`.
`output_codec="source"` remuxes H.264 or JPEG as H.264 or MJPEG, respectively.
For JPEG-to-H.264 encoding, `h264_crf` is an integer from `0` through `51`
(default `23`), and `h264_preset` is an x264 preset (default `"medium"`). These
are constructor options shared by all rows processed by that extractor.

Supported camera messages are ROS2 `sensor_msgs/msg/CompressedImage` with
H.264 data and CDR-encoded `foxglove.CompressedImage` with JPEG data. Camera
timestamps must be strictly increasing. Source H.264 with B-frame reordering
is rejected. The generated MP4 is decoded for validation before publication.

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
size. TOS output uses `temp_dir` when set, otherwise the system temporary
directory; local output uses the destination directory. Make the selected
directory writable on every worker. JPEG-to-H.264 conversion and final
validation use CPU; this UDF does not use GPU/NVENC.

The operator writes and verifies a temporary MP4 before publishing. For TOS
output it verifies a unique staging object and conditionally publishes to the
final key without overwriting an existing object. Use a TOS bucket where
versioning has **never** been enabled; enabled or suspended versioning is
rejected because it does not preserve the no-overwrite guarantee. If final
verification becomes ambiguous, the UDF preserves the final key rather than
risk deleting a concurrent writer's object; inspect that key before retrying.

Daft may retry a row, but cannot make these external writes exactly once.
Use unique run-scoped destination URIs and handle an already-existing output
as an explicit conflict. Do not rely on global row order for execution.
