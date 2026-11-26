from __future__ import annotations

import array

import pyarrow as pa
import pytest

from mmops import BatchOutputError, batch
from mmops._arrow import to_arrow_argument, validate_batch_output

_INVALID_CONTAINERS: list[object] = [
    [1, 2],
    array.array("q", [1, 2]),
    pa.record_batch({"value": [1, 2]}),
    pa.table({"value": [1, 2]}),
    pa.chunked_array([[1], [2]]),
]

try:
    import numpy as np
except ModuleNotFoundError:
    pass
else:
    _INVALID_CONTAINERS.append(np.array([1, 2]))


class SeriesLike:
    def __init__(self, array: pa.Array) -> None:
        self.array = array
        self.calls = 0

    def to_arrow(self) -> pa.Array:
        self.calls += 1
        return self.array


def _definition(dtype: pa.DataType = pa.int64()):
    @batch(return_dtype=dtype)
    def identity(values: pa.Array) -> pa.Array:
        return values

    return identity.definition


def test_arrow_argument_is_passed_through_without_copy() -> None:
    values = pa.array([1, 2, 3])

    assert to_arrow_argument(values) is values


def test_series_like_argument_uses_its_public_to_arrow_boundary() -> None:
    values = pa.array([1, 2, 3])
    series = SeriesLike(values)

    assert to_arrow_argument(series) is values
    assert series.calls == 1


def test_scalar_argument_is_passed_through() -> None:
    marker = object()

    assert to_arrow_argument(marker) is marker


@pytest.mark.parametrize("invalid", _INVALID_CONTAINERS)
def test_batch_output_rejects_non_array_containers(invalid: object) -> None:
    with pytest.raises(BatchOutputError, match="pyarrow.Array"):
        validate_batch_output(invalid, _definition(), expected_length=2)


@pytest.mark.parametrize(
    ("declared", "actual"),
    [
        (pa.string(), pa.array(["a"], type=pa.large_string())),
        (pa.large_string(), pa.array(["a"], type=pa.string())),
        (pa.binary(), pa.array([b"a"], type=pa.large_binary())),
        (pa.large_binary(), pa.array([b"a"], type=pa.binary())),
    ],
)
def test_string_and_binary_offset_widths_are_logically_compatible(
    declared: pa.DataType, actual: pa.Array
) -> None:
    assert validate_batch_output(actual, _definition(declared), 1) is actual


def test_nested_arrow_dtype_must_match_exactly() -> None:
    output = pa.array([[1.0]], type=pa.list_(pa.float64()))

    with pytest.raises(
        BatchOutputError, match=r"list<item: float>.*list<item: double>"
    ):
        validate_batch_output(
            output,
            _definition(pa.list_(pa.float32())),
            expected_length=1,
        )


def test_empty_array_satisfies_empty_batch_contract() -> None:
    output = pa.array([], type=pa.int64())

    assert validate_batch_output(output, _definition(), 0) is output


def test_batch_output_length_must_match_input_length() -> None:
    output = pa.array([1, 2], type=pa.int64())

    with pytest.raises(BatchOutputError, match="expected 3 rows.*returned 2"):
        validate_batch_output(output, _definition(), expected_length=3)
