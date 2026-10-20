# Multimodal Operator Development Guide

## Purpose

This guide defines how to develop and review operators in `daft-mmops`.
Operators currently run on Daft and turn input records into enriched 1:1
records for multimodal preprocessing, inference, and annotation.

For installation and user-facing examples, see [README.md](../README.md).
Automated contributors must also follow [AGENTS.md](../AGENTS.md).

## Choose an execution shape

| Workload | Recommended API |
|---|---|
| Lightweight independent transform | standalone `@mmops.row` |
| Arrow-vectorized transform | standalone `@mmops.batch` |
| Reusable model, tokenizer, codec, or client | `@mmops.operator` class |
| Low-latency and high-throughput forms of one model | row and batch methods on one class |
| Network-bound annotation | async row or batch method with bounded concurrency |

Use `def` or `async def` to express synchronous or asynchronous behavior. Row
or batch grain belongs on each method, not on `@mmops.operator`; one class may
therefore expose several row and batch entrypoints.

Methods without `@mmops.row` or `@mmops.batch` are internal helpers and are not
exposed to Daft.

## Public API conventions

- Do not declare an input schema on the decorator. The Python signature and
  call site bind column expressions and literal arguments.
- Row `return_dtype` is optional when Daft can infer it from the return type.
- Batch `return_dtype` is required and must be a `pyarrow.DataType`.
- Use `return_dtype`, `batch_size`, `max_retries`, and `on_error` names.
- Use `Resources(cpus=..., gpus=..., memory_bytes=...)` for resource requests.
- Put Daft-only escape hatches under `backend_options["daft"]`.

### Row template

```python
import mmops


@mmops.row
def normalize(text: str) -> str:
    return text.strip().lower()
```

Add an explicit Arrow dtype when the Python annotation cannot express the
required precision, fixed size, nullability, or struct layout:

```python
import pyarrow as pa


@mmops.row(return_dtype=pa.list_(pa.float32()))
def embed_one(text: str) -> list[float]:
    return model.embed(text)
```

### Batch template

```python
import pyarrow as pa
import pyarrow.compute as pc

import mmops


@mmops.batch(return_dtype=pa.int64(), batch_size=256)
def text_length(text: pa.Array) -> pa.Array:
    return pc.cast(pc.utf8_length(text), pa.int64())
```

Batch parameters are one or more `pa.Array` values plus ordinary literals.
Batch output must be exactly one `pa.Array` with the declared dtype.

### Stateful template

```python
import pyarrow as pa

import mmops


@mmops.operator(
    resources=mmops.Resources(gpus=1),
    max_concurrency=4,
)
class Embedder:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    @mmops.row(return_dtype=pa.list_(pa.float32()))
    async def embed_one(self, text: str) -> list[float]:
        return await self.model.embed(text)

    @mmops.batch(
        return_dtype=pa.list_(pa.float32()),
        batch_size=64,
    )
    async def embed_batch(self, text: pa.Array) -> pa.Array:
        return await self.model.embed_batch(text)
```

Initialize models and clients in `__init__`. Daft lazily creates and reuses the
instance in a worker/actor. Constructor validation and initialization errors
therefore appear when an entrypoint is first invoked, not when the client-side
operator handle is created.

## Batch data contract

Version 1 has strict 1:1 cardinality: every input row produces one output row.

Batch entrypoints must:

- accept `pa.Array`, not `daft.Series`;
- return `pa.Array` or its `pa.StructArray` subtype;
- return the same number of rows as the input arrays;
- return the exact declared dtype;
- preserve documented null semantics;
- handle empty and final short batches;
- avoid whole-batch `to_pylist()` conversion when Arrow kernels or buffer
  interfaces can perform the operation.

Do not return `pa.Table`, `pa.RecordBatch`, `pa.ChunkedArray`, Pandas objects,
NumPy arrays, Python lists, or `daft.Series`. If a model or external service
requires Python objects, document the conversion and its expected cost.

The framework treats top-level `string`/`large_string` and
`binary`/`large_binary` as logically compatible because Daft may normalize
Arrow offset widths. Nested types and all other dtypes must match exactly.

### Multiple output columns

One UDF invocation returns one logical Arrow array. Use a struct dtype and
`unnest=True` for several named output columns:

```python
result_type = pa.struct(
    [
        pa.field("embedding", pa.list_(pa.float32())),
        pa.field("score", pa.float32()),
    ]
)


@mmops.batch(return_dtype=result_type, unnest=True)
def analyze(text: pa.Array) -> pa.StructArray:
    embeddings, scores = run_model(text)
    return pa.StructArray.from_arrays(
        [embeddings, scores],
        fields=list(result_type),
    )
```

Treat field names and types as public API. Names must describe the computed
value accurately.

## Row and batch consistency

When the same capability has row and batch entrypoints, align:

- output dtype and field names;
- null and empty-input behavior;
- error behavior;
- literal option defaults;
- ordering guarantees.

Do not rely on global row order for asynchronous Daft execution unless the
operator explicitly guarantees it.

## State semantics

Operator state is local to one Daft worker/actor instance. It is suitable for
expensive initialization and connection reuse, not for:

- cross-worker shared state;
- keyed routing or ordering;
- checkpoint/restore;
- exactly-once processing;
- a guaranteed teardown callback.

Keep credentials, access tokens, and environment-specific endpoints out of
source code and serialized operator metadata.

Operators that publish files or objects are still 1:1 row transforms, but the
external write is a side effect. They must write to a temporary target, verify
the completed artifact, publish without silently overwriting an existing
target, and return only after publication succeeds. Their documentation must
state retry/idempotency behavior and temporary-disk requirements; callers must
use unique run-scoped output paths when the engine cannot provide exactly-once
external writes.

For object stores that cannot atomically upload and verify a final key, publish
through a unique staging key: upload, verify the staged size and content
digest, conditionally copy to the final no-overwrite key, verify the final
size and digest, and remove the staging object. A failed staging verification
must never poison the final key.
An operator may delegate those backend-specific steps to an injected
filesystem or publication wrapper, but must check for that capability before
reading the input and fail closed when it is unavailable.
Calculate the local content digest while streaming the upload instead of
adding a separate full local-file pass. A wrapper may return a publication
receipt that attests the final object's size and cryptographic digest. Skip a
final-object download only when the digest is a backend-provided cryptographic
checksum bound to that exact final object or version; ETags and caller-written
metadata are not sufficient. Otherwise perform one bounded final readback and
verify its exact size and digest.

Before relying on an object-store no-overwrite condition, verify that bucket
features such as versioning do not disable that condition. Fail closed when
the store cannot guarantee create-only publication. Never delete an ambiguous
unversioned final key after failed verification because a concurrent writer
may have replaced it between HEAD and DELETE; only an exact returned version
identifier is safe to remove.
Likewise, do not delete a staging object by key when its identity check has
failed; warn and leave it for explicit inspection instead.

## Daft integration boundary

Operator business code uses Python scalars and PyArrow arrays. Daft integration
is implemented centrally in `src/mmops/backends/daft.py` with public APIs only:

- `daft.func` and `daft.func.batch`;
- `daft.cls`;
- `daft.method` and `daft.method.batch`.

Do not use private APIs such as `Func._from_func`, `mark_cls_method`, or
`daft.udf.udf_v2` internals. Never silently discard an explicitly requested
execution option; raise `DaftCompatibilityError` when the installed Daft
version cannot represent it.

Definitions and constructed operator handles must remain pickleable across
Daft worker boundaries without serializing locks or client-side caches.

The current release executes only through Daft. Keep execution bindings
isolated internally so another engine can be evaluated later, but do not expose
or document an unimplemented engine as supported.

## Package and dependency organization

Reusable operators live under `src/mmops/operators/`. As the catalog grows,
group cohesive capabilities by domain:

```text
src/mmops/operators/
├── text/
├── image/
├── audio/
├── video/
├── robotics/
├── document/
└── multimodal/
```

Do not add empty directory scaffolding. Create a domain package when real
operators form a useful group.

Keep the core package lightweight. Heavy model, media, codec, or service SDKs
belong in optional extras and should be imported only by the relevant domain
module or execution path. Users must not need to rebuild Daft or edit installed
Daft source code.

The root `mmops.operators` catalog lazily exports public operators from their
domain packages. Adding a root export must not eagerly import any domain
package; importing one domain must not import another. Domain modules must also
keep optional heavy dependencies on the execution paths that require them.

Each planned domain has a same-named extra in `pyproject.toml`. Extras for
domains without production operators remain empty and are documented as
reserved, not supported. When the first operator is released, add the Daft
runtime and any domain dependencies to that extra. The `all` extra must equal
the flattened dependency union of `text`, `image`, `audio`, `video`,
`robotics`, `document`, and `multimodal`; PEP 621 extras cannot reference one
another by name. Update the domain extra, `all`, README installation table,
and packaging tests in the same change as a new operator dependency.

Each production operator must document:

- accepted input representation;
- output dtype and field meanings;
- null and failure behavior;
- optional dependencies, model, or service requirements;
- CPU, GPU, memory, batch-size, and concurrency expectations;
- runtime credentials or configuration without exposing their values.

## Testing requirements

Every new operator needs:

- eager Python or Arrow tests;
- a real Daft DataFrame integration test;
- null behavior coverage;
- empty input coverage when applicable;
- final-short-batch coverage when `batch_size` is set;
- output dtype and field-name assertions;
- row/batch consistency coverage when both forms exist;
- error-path coverage for the public contract.

Run the repository verification commands listed in
[AGENTS.md](../AGENTS.md#tests-and-verification) before completion.

## Review checklist

- The operator solves a reusable multimodal processing or annotation task.
- Its public name and output fields accurately describe its behavior.
- Row and batch signatures follow the scalar/Arrow contract.
- Batch output has the declared dtype and preserves 1:1 cardinality.
- Null, empty, error, and ordering semantics are explicit and tested.
- Expensive resources initialize in `__init__`, not at module import.
- Optional dependencies do not affect unrelated imports.
- Secrets are supplied at runtime and are not logged or serialized.
- Resource and concurrency expectations are documented.
- Public exports, README/operator docs, and tests change together.
