from __future__ import annotations

import io

import daft
import pyarrow as pa
import pytest

import mmops
from mmops.backends.daft import _namespaced_options


def test_struct_batch_output_is_unnested_into_named_columns() -> None:
    return_dtype = pa.struct(
        [
            pa.field("length", pa.int64()),
            pa.field("upper", pa.string()),
        ]
    )

    @mmops.batch(return_dtype=return_dtype, unnest=True)
    def features(text: pa.Array) -> pa.StructArray:
        values = text.to_pylist()
        return pa.StructArray.from_arrays(
            [
                pa.array([None if value is None else len(value) for value in values]),
                pa.array(
                    [None if value is None else value.upper() for value in values]
                ),
            ],
            fields=list(return_dtype),
        )

    df = daft.from_pydict({"text": ["hi", None, "world"]})
    result = df.select(features(df["text"])).to_pydict()

    assert result == {
        "length": [2, None, 5],
        "upper": ["HI", None, "WORLD"],
    }


def test_batch_output_must_be_an_arrow_array() -> None:
    @mmops.batch(return_dtype=pa.int64())
    def invalid(values: pa.Array) -> list[int]:
        return values.to_pylist()

    with pytest.raises(mmops.BatchOutputError, match="pyarrow.Array"):
        invalid(pa.array([1, 2]))


def test_batch_output_dtype_is_checked_during_eager_evaluation() -> None:
    @mmops.batch(return_dtype=pa.float32())
    def invalid(values: pa.Array) -> pa.Array:
        return pa.array(values.to_pylist(), type=pa.float64())

    with pytest.raises(mmops.BatchOutputError, match="float.*double"):
        invalid(pa.array([1.0]))


def test_batch_output_cardinality_is_checked_during_eager_evaluation() -> None:
    @mmops.batch(return_dtype=pa.int64())
    def invalid(values: pa.Array) -> pa.Array:
        return pa.array([1])

    with pytest.raises(mmops.BatchOutputError, match="expected 2 rows.*returned 1"):
        invalid(pa.array([1, 2]))


def test_explicit_unsupported_daft_option_is_never_silently_dropped() -> None:
    @mmops.row(backend_options={"daft": {"not_a_real_option": True}})
    def identity(value: int) -> int:
        return value

    with pytest.raises(
        mmops.DaftCompatibilityError,
        match="not_a_real_option.*daft.func",
    ):
        identity(1)


def test_explicit_resource_unsupported_by_installed_daft_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_func = daft.func

    def func_without_cpus(fn: object, *, return_dtype: object = None) -> object:
        return original_func(fn, return_dtype=return_dtype)

    monkeypatch.setattr(daft, "func", func_without_cpus)

    @mmops.row(resources=mmops.Resources(cpus=1))
    def identity(value: int) -> int:
        return value

    with pytest.raises(
        mmops.DaftCompatibilityError,
        match="cpus.*daft.func",
    ):
        identity(1)


def test_invalid_daft_option_namespace_is_reported() -> None:
    @mmops.row(backend_options={"daft": "invalid"})
    def identity(value: int) -> int:
        return value

    with pytest.raises(mmops.DaftCompatibilityError, match="must be a mapping"):
        identity(1)


def test_backend_options_preserve_nested_container_types() -> None:
    original = {
        "daft": {
            "payload": {
                "list": [1, 2],
                "tuple": (3, 4),
                "set": {5, 6},
                "frozenset": frozenset({7, 8}),
            }
        }
    }

    @mmops.row(backend_options=original)
    def identity(value: int) -> int:
        return value

    forwarded = _namespaced_options(identity.definition.backend_options)["payload"]

    assert isinstance(forwarded["list"], list)
    assert isinstance(forwarded["tuple"], tuple)
    assert isinstance(forwarded["set"], set)
    assert isinstance(forwarded["frozenset"], frozenset)


def test_backend_options_cannot_override_entrypoint_options() -> None:
    @mmops.batch(
        return_dtype=pa.int64(),
        batch_size=16,
        backend_options={"daft": {"batch_size": 1}},
    )
    def identity(values: pa.Array) -> pa.Array:
        return values

    with pytest.raises(
        mmops.DaftCompatibilityError,
        match="batch_size.*top-level",
    ):
        identity(pa.array([1]))


def test_backend_options_cannot_override_operator_options() -> None:
    @mmops.operator(
        max_concurrency=4,
        backend_options={"daft": {"max_concurrency": 1}},
    )
    class Identity:
        @mmops.row
        def run(self, value: int) -> int:
            return value

    with pytest.raises(
        mmops.DaftCompatibilityError,
        match="max_concurrency.*top-level",
    ):
        Identity()


def test_supported_operator_options_reach_daft_plan() -> None:
    @mmops.operator(max_concurrency=2, name="Scorer")
    class Scorer:
        @mmops.row
        async def score(self, value: int) -> int:
            return value

    scorer = Scorer()
    df = daft.from_pydict({"value": [1]}).select(scorer.score(daft.col("value")))

    output = io.StringIO()
    df.explain(file=output, show_all=True)
    plan = output.getvalue()

    assert "Scorer" in plan
    assert "concurrency = 2" in plan


def test_unset_gpu_option_is_not_forwarded_to_daft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_cls = daft.cls

    def cls_without_gpu_option(class_: type) -> type:
        return original_cls(class_)

    monkeypatch.setattr(daft, "cls", cls_without_gpu_option)

    @mmops.operator
    class Identity:
        @mmops.row
        def run(self, value: int) -> int:
            return value

    assert Identity() is not None
