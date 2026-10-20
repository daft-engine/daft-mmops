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
| `robotics` | Daft runtime, MCAP compressed-image decoding, H.264/JPEG processing, and fsspec interfaces; object-store backends are caller-supplied |
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

`McapVideoExtractor` is a Daft row operator that converts a compressed MCAP
camera stream to MP4. It accepts local or HTTP(S) input and injected
fsspec-compatible filesystems for remote input and safe remote output. See the
[operator guide](docs/operators/robotics/mcap-video-extractor.md) for its
filesystem contract, output schema, and publication safeguards.

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
