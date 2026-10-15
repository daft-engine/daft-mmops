# daft-mmops

`daft-mmops` is a multimodal data processing operator library for Daft. It is
intended for reusable text, image, audio, video, robotics, document, and
cross-modal preprocessing, enrichment, model inference, and data annotation.

The current release runs on Daft and provides:

- a growing operator catalog under `mmops.operators`;
- `@mmops.row`, `@mmops.batch`, and `@mmops.operator` for developing operators;
- Python-scalar row processing and PyArrow-native batch processing;
- stateless/stateful and synchronous/asynchronous execution;
- Daft resource, concurrency, retry, and error-policy integration.

For implementation rules and complete templates, see the
[Multimodal Operator Development Guide](docs/operator-development.md).
Repository-wide automation rules are defined in [AGENTS.md](AGENTS.md).

## Installation

Install only the released domain you need:

```bash
python -m pip install 'daft-mmops[text]'
```

Available extras:

| Extra | Purpose |
|---|---|
| `text` | Daft runtime and all released text operators |
| `image` | reserved for image operators; currently empty |
| `audio` | reserved for audio operators; currently empty |
| `video` | reserved for video operators; currently empty |
| `robotics` | Daft runtime, MCAP/ROS2 decoding, H.264 remuxing, and fsspec TOS support |
| `document` | reserved for document operators; currently empty |
| `multimodal` | reserved for cross-modal operators; currently empty |
| `all` | dependency union of `text`, `image`, `audio`, `video`, `robotics`, `document`, and `multimodal` |
| `daft` | Daft runtime for SDK users who only define custom operators |
| `test` | dependencies needed to run pytest |
| `dev` | complete contributor toolchain: tests, formatting, build, and metadata checks |

From a source checkout, add `-e`, for example
`python -m pip install -e '.[text]'`. Python 3.10 or newer and PyArrow 12 or
newer are required. Version `0.1` targets the public Daft 0.7 UDF API.

All planned domain names are reserved as installation extras. An empty extra
does not mean that the domain is implemented; it provides a stable install
name until its first production operator adds runtime dependencies. The `all`
extra is always the flattened dependency union of all seven domain extras, so
empty domains currently contribute no dependencies.

## Quick start

Built-in operators are ordinary Daft expressions:

```python
import daft

from mmops.operators import KeywordMatcher, normalize_text

df = daft.from_pydict(
    {
        "text": ["  ROBOT navigation  ", None, "manipulation"],
    }
)

normalized = df.with_column(
    "normalized",
    normalize_text(df["text"], lowercase=True),
)
matcher = KeywordMatcher(
    ["robot", "navigation", "manipulation"],
    whole_word=True,
)
result = normalized.with_column(
    "keywords",
    matcher.match_batch(normalized["normalized"]),
)
```

Define a lightweight row operator with Python scalar types:

```python
import mmops


@mmops.row
def normalize(text: str) -> str:
    return text.strip().lower()


df = df.with_column("normalized", normalize(df["text"]))
```

Batch and stateful operators use the same expression style. See the
[development guide](docs/operator-development.md#choose-an-execution-shape) for
Arrow batch and model-backed class templates.

## Text operators

The first text package contains one stateless and one stateful operator:

| Operator | State and entrypoints | Behavior | Framework contract covered |
|---|---|---|---|
| `normalize_text` | stateless batch | Unicode, lowercase, whitespace collapse, and trim options | scalar options, Arrow kernels, null and empty batches |
| `KeywordMatcher` | worker-local row + batch | compiled keyword matching with case and whole-word options | constructor options, row/batch parity, Arrow list output, null and pickle |

```python
from mmops.operators.text import KeywordMatcher, normalize_text

df = df.with_column(
    "normalized",
    normalize_text(df["text"], lowercase=True),
)

matcher = KeywordMatcher(
    ["robot", "manipulation", "navigation"],
    case_sensitive=False,
    whole_word=True,
)
df = df.with_column("keywords", matcher.match_batch(df["normalized"]))
```

`normalize_text` uses Arrow compute kernels and preserves nulls.
`KeywordMatcher` compiles escaped regex patterns once per Daft worker/actor,
returns each configured keyword at most once in configuration order, and
returns null for null text. Keyword configuration is validated when the first
entrypoint is invoked because Daft initializes operator instances lazily.

## Robotics operators

`McapVideoExtractor` turns one MCAP camera capture into one MP4 object. MCAP
input supports local paths, `tos://bucket/key` object URIs, and `http://` or
`https://` URLs. MP4 output supports local paths and TOS object URIs; HTTP is
read-only. HTTP responses and TOS objects are consumed as non-seeking streams,
so the UDF does not load the complete MCAP into memory. Inputs may contain
either ROS2 `sensor_msgs/msg/CompressedImage` H.264 messages or CDR-encoded
`foxglove.CompressedImage` JPEG messages. Camera timestamps must be strictly
increasing. Source H.264 must not use B-frame reordering. The completed MP4 is
fully decoded once for validation, while the UDF keeps the large MP4 payload
out of the Daft row and returns only metadata.

```python
import daft

from mmops.operators.robotics import McapVideoExtractor

df = daft.from_pydict(
    {
        "mcap_uri": ["tos://input-bucket/captures/cam0.mcap"],
        "mp4_uri": ["tos://output-bucket/run-20261012/cam0.mp4"],
    }
)
extractor = McapVideoExtractor()  # defaults to H.264 output
result = df.with_column(
    "video",
    extractor.extract(df["mcap_uri"], df["mp4_uri"]),
)
```

`output_codec` is a batch-level constructor policy:

| `output_codec` | H.264 input | JPEG input |
| --- | --- | --- |
| `"h264"` (default) | direct H.264 remux | decode JPEG and encode with CPU `libx264` |
| `"source"` | direct H.264 remux | direct MJPEG remux |

One extractor instance uses one output policy for the whole batch. Configure
JPEG-to-H.264 quality with `h264_crf` (integer `0` through `51`, default `23`)
and `h264_preset` (an x264 preset, default `"medium"`). H.264 encoding produces
YUV420P output. The H.264 default prioritizes downstream storage and access
performance, at the cost of CPU encoding for JPEG inputs. Set
`output_codec="source"` explicitly to avoid re-encoding JPEG. Direct remux
copies the compressed frame payloads without re-encoding; validation decoding
does not alter the output.

The result struct contains `output_uri`, `topic`, `frame_count`, `duration_ns`,
`size_bytes`, `sha256`, and `codec`. `duration_ns` is the finalized MP4 timeline
duration, including the last frame duration inferred from the preceding frame;
`codec` is the actual MP4 codec (`h264` or `mjpeg`). A single supported camera
topic is selected automatically; pass `topic="/camera/image/compressed"` when
the MCAP contains more than one supported topic. Paired null input/output rows
return null. A partial null pair, unsupported requested format, non-increasing
timestamps, unsupported source H.264 B-frame reordering, undecodable video,
missing topic, or an existing output is an error.

TOS access uses the `tosfs` fsspec filesystem with `TOS_ENDPOINT`,
`TOS_REGION`, `TOS_ACCESS_KEY`, `TOS_SECRET_KEY`, and optional
`TOS_SECURITY_TOKEN` from the worker environment. The robotics extra declares
`fsspec` and `tosfs`, without a direct `tos` requirement; `tosfs` manages its
own SDK dependency. Environment-specific values are not constructor arguments
and are therefore not serialized with the operator.
Each output is first completed and verified in a worker-local temporary file;
set `temp_dir` when the default temporary filesystem is too small. Allow free
temporary disk roughly equal to the resulting MP4 size. TOS publishing first
uploads and verifies a unique staging object, then performs a conditional
server-side copy to the final key; local publishing uses a no-overwrite atomic
link. TOS final-key publication fails closed when bucket versioning is enabled
or suspended because TOS ignores no-overwrite headers for versioned buckets.
Use a bucket where versioning has never been enabled and unique output URIs for
each run because Daft retries do not provide exactly-once semantics for
external side effects.

JPEG-to-H.264 mode also needs CPU for decode and encode; size and throughput
depend on CRF, preset, resolution, frame rate, and scene complexity. It does
not use GPU/NVENC. Use direct `source` mode when preserving compressed JPEG
payloads and minimizing CPU are more important than file size or H.264 player
compatibility.

## Current execution scope

Version 1 supports strict 1:1 Python UDF execution on Daft:

| | row | batch |
|---|---|---|
| stateless sync | supported | supported |
| stateless async | supported | supported |
| stateful sync | supported | supported |
| stateful async | supported | supported |

Batch functions accept one or more `pa.Array` inputs and return one `pa.Array`.
Use `pa.StructArray` with `unnest=True` for multiple output columns. Daft owns
planning, partitioning, local or distributed execution configured through
Daft, resource scheduling, async concurrency, retry handling, and backpressure.

Generator/flat-map UDFs, UDAFs, dynamic schemas, distributed strong state, and
execution directly through other data engines are outside version 1.

## Development

Run the formal text pipeline example:

```bash
python examples/text_pipeline.py
```

Install the contributor environment and run the test suite:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Before adding or reviewing an operator, read
[docs/operator-development.md](docs/operator-development.md) for the data
contract, state semantics, dependency policy, test matrix, and review checklist.
