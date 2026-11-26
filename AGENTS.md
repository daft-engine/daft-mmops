# AGENTS.md

## Scope and source of truth

These instructions apply to the entire repository.

- [README.md](README.md) is the user-facing source for installation, current
  capabilities, operators, and examples.
- [docs/operator-development.md](docs/operator-development.md) is the source of
  truth for operator APIs, data contracts, state semantics, dependencies,
  testing requirements, and review criteria.
- This file contains execution rules for automated contributors. Do not copy
  the detailed development guide back into this file.

## Non-negotiable project constraints

- The product is a multimodal data processing and annotation operator library.
- The current release executes operators only through Daft.
- Do not claim that Ray Data or another engine is implemented or supported.
- Keep engine bindings isolated internally so future implementations do not
  require rewriting every operator.
- Keep the planned domain extras declared. Unreleased domains remain
  empty. `all` must be the exact flattened dependency union of `text`, `image`,
  `audio`, `video`, `robotics`, `document`, and `multimodal`; update it,
  installation docs, and packaging tests whenever any domain dependency
  changes.
- Use only public Daft APIs. Never import private `daft.udf.udf_v2` helpers such
  as `Func._from_func` or `mark_cls_method`.
- Operator business code must use Python scalar or PyArrow interfaces, not
  `daft.Series` or `daft.DataType`.
- Version 1 has strict 1:1 cardinality.
- Never silently drop an explicitly requested execution option.
- Keep secrets and environment-specific credentials out of code, logs, and
  serialized metadata.

## Change workflow

Before editing:

1. Read the relevant source, tests, README section, and
   `docs/operator-development.md`.
2. Confirm the change belongs in the operator catalog, public SDK, or Daft
   integration layer.
3. Preserve unrelated user changes and keep the diff scoped to the request.
4. Work on a feature branch, not directly on `main`.

While implementing:

1. Add a focused failing test and verify the expected failure.
2. Implement the minimum behavior needed to pass it.
3. Keep reusable operators under `src/mmops/operators/` and Daft binding logic
   under `src/mmops/backends/daft.py`.
4. Follow the API, Arrow ABI, state, dependency, null, schema, and test rules in
   [the development guide](docs/operator-development.md).
5. Update public exports and user/developer documentation in the same change as
   public behavior.

Do not add speculative runtimes, schedulers, state stores, empty domain
packages, aliases, or compatibility layers without a concrete requirement.

## Review gates

Reject or revise a change when any of the following is true:

- a field name does not accurately describe the computed value;
- analogous row and batch paths differ in dtype, null, error, or field
  semantics without documentation;
- batch output is not a `pa.Array`/`pa.StructArray`, has the wrong dtype, or
  changes row count;
- an avoidable whole-batch Python-object conversion replaces an Arrow-native
  path;
- a model/client initializes at module import rather than in operator
  `__init__`;
- an optional heavy dependency breaks unrelated imports;
- private Daft APIs are used;
- worker-bound definitions or handles are no longer pickleable;
- documentation describes unimplemented execution support;
- tests cover only helpers or mocks instead of real operator behavior.

## Tests and verification

Follow the test matrix in
[docs/operator-development.md](docs/operator-development.md#testing-requirements).
Do not assume global row order for asynchronous Daft execution unless the
operator contract guarantees it.

Before claiming completion, run from the repository root:

```bash
python -m pip install -e '.[dev]'
pytest -q
black --check src tests examples
python -m compileall -q src tests examples
python -m build
python -m twine check dist/*
git diff --check
```

When package contents or worker serialization changes, install the built wheel
outside the repository and run a real Daft smoke test. Distinguish inherited
environment dependency problems from mmops failures.

## Documentation and delivery

- Keep README concise and user-facing; link to the development guide for
  authoring details.
- Keep detailed operator rules in `docs/operator-development.md` rather than
  duplicating them here or in README.
- Describe only implemented behavior as supported.
- Do not publish reference/demo functions in the operator catalog. Keep
  demonstrations under `examples/` or `tests/`.
- Confirm repository, branch, target remote, and clean status before push.
- Do not use destructive git commands.
- PR summaries must state behavior, schema, dependencies, Daft execution scope,
  and exact verification evidence.
