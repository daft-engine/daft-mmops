from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence, TypeVar

import pyarrow as pa

ErrorPolicy = Literal["raise", "log", "ignore"]
EntrypointKind = Literal["row", "batch"]
K = TypeVar("K")
V = TypeVar("V")


class MmopsError(Exception):
    """Base class for public mmops errors."""


class DefinitionError(MmopsError, ValueError):
    """Raised when an operator definition is invalid."""


class BatchOutputError(MmopsError, ValueError):
    """Raised when a batch callable violates its Arrow output contract."""


class DaftCompatibilityError(MmopsError, RuntimeError):
    """Raised when the installed Daft cannot represent requested options."""


class FrozenDict(Mapping[K, V]):
    """A pickleable, recursively immutable mapping."""

    def __init__(self, value: Mapping[K, V] | None = None) -> None:
        self._data = {key: _freeze(nested) for key, nested in (value or {}).items()}

    def __getitem__(self, key: K) -> V:
        return self._data[key]  # type: ignore[return-value]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __reduce__(self) -> tuple[type[FrozenDict[Any, Any]], tuple[dict[Any, Any]]]:
        return FrozenDict, (self._data,)


class FrozenList(Sequence[V]):
    """An immutable sequence that records an original list value."""

    def __init__(self, value: Sequence[V]) -> None:
        self._data = tuple(_freeze(item) for item in value)

    def __getitem__(self, index: int | slice) -> V | tuple[V, ...]:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenList({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return tuple(self) == tuple(other)
        return False

    def __reduce__(self) -> tuple[type[FrozenList[Any]], tuple[tuple[Any, ...]]]:
        return FrozenList, (self._data,)


class FrozenSet(frozenset[V]):
    """An immutable set that records an original mutable set value."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, list):
        return FrozenList(value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return FrozenSet(_freeze(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return FrozenDict(value)


def _validate_resource_number(name: str, value: object | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DefinitionError(f"{name} must be a finite non-negative number")
    if not math.isfinite(float(value)) or value < 0:
        raise DefinitionError(f"{name} must be a finite non-negative number")


def _validate_non_negative_int(name: str, value: object | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DefinitionError(f"{name} must be a non-negative integer, got {value!r}")


def _validate_positive_int(name: str, value: object | None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DefinitionError(f"{name} must be a positive integer, got {value!r}")


def _validate_on_error(value: str | None) -> None:
    if value not in (None, "raise", "log", "ignore"):
        raise DefinitionError(
            "on_error must be one of 'raise', 'log', or 'ignore', " f"got {value!r}"
        )


@dataclass(frozen=True)
class Resources:
    """Resource requirements for one operator instance."""

    cpus: float | None = None
    gpus: float | None = None
    memory_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_resource_number("cpus", self.cpus)
        _validate_resource_number("gpus", self.gpus)
        _validate_non_negative_int("memory_bytes", self.memory_bytes)


@dataclass(frozen=True)
class EntrypointDefinition:
    """Definition of one row or batch entrypoint."""

    original: Callable[..., Any]
    kind: EntrypointKind
    return_dtype: pa.DataType | None = None
    resources: Resources = field(default_factory=Resources)
    unnest: bool = False
    batch_size: int | None = None
    max_retries: int | None = None
    on_error: ErrorPolicy | None = None
    use_process: bool | None = None
    backend_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.return_dtype is not None and not isinstance(
            self.return_dtype, pa.DataType
        ):
            raise DefinitionError(
                "return_dtype must be a pyarrow.DataType, "
                f"got {type(self.return_dtype).__name__}"
            )
        if self.kind == "batch" and self.return_dtype is None:
            raise DefinitionError("batch entrypoints require return_dtype")
        if self.kind == "row" and self.batch_size is not None:
            raise DefinitionError("batch_size is only valid for batch entrypoints")
        if self.batch_size is not None:
            _validate_positive_int("batch_size", self.batch_size)
        _validate_non_negative_int("max_retries", self.max_retries)
        _validate_on_error(self.on_error)
        object.__setattr__(
            self, "backend_options", _frozen_mapping(self.backend_options)
        )


@dataclass(frozen=True)
class OperatorDefinition:
    """Definition of a stateful operator class."""

    original: type
    entrypoints: Mapping[str, EntrypointDefinition]
    resources: Resources = field(default_factory=Resources)
    max_concurrency: int | None = None
    max_retries: int | None = None
    on_error: ErrorPolicy | None = None
    use_process: bool | None = None
    name: str | None = None
    backend_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entrypoints:
            raise DefinitionError(
                f"operator {self.original.__name__!r} must define at least one "
                "@mmops.row or @mmops.batch entrypoint"
            )
        if self.max_concurrency is not None:
            _validate_positive_int("max_concurrency", self.max_concurrency)
        _validate_non_negative_int("max_retries", self.max_retries)
        _validate_on_error(self.on_error)
        object.__setattr__(self, "entrypoints", _frozen_mapping(self.entrypoints))
        object.__setattr__(
            self, "backend_options", _frozen_mapping(self.backend_options)
        )
