from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any

import pyarrow as pa

from .._arrow import to_arrow_argument, validate_batch_output
from .._definitions import (
    DaftCompatibilityError,
    EntrypointDefinition,
    FrozenList,
    FrozenSet,
    OperatorDefinition,
)

_ENTRYPOINT_OPTION_KEYS = frozenset(
    {
        "return_dtype",
        "unnest",
        "cpus",
        "gpus",
        "memory_bytes",
        "batch_size",
        "max_retries",
        "on_error",
        "use_process",
    }
)
_OPERATOR_OPTION_KEYS = frozenset(
    {
        "cpus",
        "gpus",
        "memory_bytes",
        "use_process",
        "max_concurrency",
        "max_retries",
        "on_error",
        "name_override",
    }
)


def _arrow_return_dtype(dtype: Any) -> Any:
    import daft

    return None if dtype is None else daft.DataType.from_arrow_type(dtype)


def _namespaced_options(options: Mapping[str, Any]) -> dict[str, Any]:
    value = options.get("daft", {})
    if not isinstance(value, Mapping):
        raise DaftCompatibilityError(
            "backend_options['daft'] must be a mapping of Daft decorator options"
        )
    return {key: _thaw(nested) for key, nested in value.items()}


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(nested) for key, nested in value.items()}
    if isinstance(value, FrozenList):
        return [_thaw(item) for item in value]
    if isinstance(value, FrozenSet):
        return {_thaw(item) for item in value}
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_thaw(item) for item in value)
    return value


def _merge_backend_options(
    options: dict[str, Any],
    backend_options: Mapping[str, Any],
    reserved_keys: frozenset[str],
) -> None:
    daft_options = _namespaced_options(backend_options)
    conflicts = sorted(reserved_keys.intersection(daft_options))
    if conflicts:
        joined = ", ".join(conflicts)
        raise DaftCompatibilityError(
            "backend_options['daft'] cannot override "
            f"{joined}; use the top-level mmops options"
        )
    options.update(daft_options)


def _supported_options(
    decorator: Callable[..., Any],
    options: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    parameters = inspect.signature(decorator).parameters
    unsupported = sorted(key for key in options if key not in parameters)
    if unsupported:
        joined = ", ".join(unsupported)
        raise DaftCompatibilityError(
            f"installed Daft does not support {joined} on {context}"
        )
    return dict(options)


def _entrypoint_options(definition: EntrypointDefinition) -> dict[str, Any]:
    options: dict[str, Any] = {
        "return_dtype": _arrow_return_dtype(definition.return_dtype),
    }
    if definition.unnest:
        options["unnest"] = True
    if definition.resources.cpus is not None:
        options["cpus"] = definition.resources.cpus
    if definition.resources.gpus is not None:
        options["gpus"] = definition.resources.gpus
    if definition.resources.memory_bytes is not None:
        options["memory_bytes"] = definition.resources.memory_bytes
    if definition.batch_size is not None:
        options["batch_size"] = definition.batch_size
    if definition.max_retries is not None:
        options["max_retries"] = definition.max_retries
    if definition.on_error is not None:
        options["on_error"] = definition.on_error
    if definition.use_process is not None:
        options["use_process"] = definition.use_process
    _merge_backend_options(
        options,
        definition.backend_options,
        _ENTRYPOINT_OPTION_KEYS,
    )
    return {key: value for key, value in options.items() if value is not None}


def _convert_batch_arguments(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any], int]:
    converted_args = tuple(to_arrow_argument(value) for value in args)
    converted_kwargs = {key: to_arrow_argument(value) for key, value in kwargs.items()}
    arrays = [
        value
        for value in (*converted_args, *converted_kwargs.values())
        if isinstance(value, pa.Array)
    ]
    if not arrays:
        raise ValueError(
            "a batch entrypoint requires at least one Arrow array argument"
        )
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError(
            "batch entrypoint Arrow arguments must have equal lengths, "
            f"got {sorted(lengths)}"
        )
    return converted_args, converted_kwargs, lengths.pop()


def _batch_bridge(definition: EntrypointDefinition) -> Callable[..., Any]:
    original = definition.original
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_bridge(*args: Any, **kwargs: Any) -> Any:
            converted_args, converted_kwargs, expected_length = (
                _convert_batch_arguments(args, kwargs)
            )
            output = await original(*converted_args, **converted_kwargs)
            return validate_batch_output(output, definition, expected_length)

        return async_bridge

    @functools.wraps(original)
    def sync_bridge(*args: Any, **kwargs: Any) -> Any:
        converted_args, converted_kwargs, expected_length = _convert_batch_arguments(
            args, kwargs
        )
        output = original(*converted_args, **converted_kwargs)
        return validate_batch_output(output, definition, expected_length)

    return sync_bridge


def compile_function(definition: EntrypointDefinition) -> Callable[..., Any]:
    """Compile one standalone definition through Daft public APIs."""

    import daft

    options = _entrypoint_options(definition)
    if definition.kind == "batch":
        decorator = daft.func.batch
        options = _supported_options(decorator, options, "daft.func.batch")
        return decorator(**options)(_batch_bridge(definition))

    decorator = daft.func
    options = _supported_options(decorator, options, "daft.func")
    return decorator(definition.original, **options)


def _compile_method(definition: EntrypointDefinition) -> Callable[..., Any]:
    import daft

    options = _entrypoint_options(definition)
    if definition.kind == "batch":
        decorator = daft.method.batch
        options = _supported_options(decorator, options, "daft.method.batch")
        return decorator(**options)(_batch_bridge(definition))

    decorator = daft.method
    options = _supported_options(decorator, options, "daft.method")
    return decorator(definition.original, **options)


def _operator_options(definition: OperatorDefinition) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if definition.resources.cpus is not None:
        options["cpus"] = definition.resources.cpus
    if definition.resources.gpus is not None:
        options["gpus"] = definition.resources.gpus
    if definition.resources.memory_bytes is not None:
        options["memory_bytes"] = definition.resources.memory_bytes
    if definition.use_process is not None:
        options["use_process"] = definition.use_process
    if definition.max_concurrency is not None:
        options["max_concurrency"] = definition.max_concurrency
    if definition.max_retries is not None:
        options["max_retries"] = definition.max_retries
    if definition.on_error is not None:
        options["on_error"] = definition.on_error
    if definition.name is not None:
        options["name_override"] = definition.name
    _merge_backend_options(
        options,
        definition.backend_options,
        _OPERATOR_OPTION_KEYS,
    )
    return options


def compile_operator(definition: OperatorDefinition) -> type:
    """Compile a stateful definition through Daft public APIs."""

    import daft

    namespace = {
        "__module__": definition.original.__module__,
        "__doc__": definition.original.__doc__,
        "__mmops_original__": definition.original,
    }
    namespace.update(
        {
            name: _compile_method(entrypoint)
            for name, entrypoint in definition.entrypoints.items()
        }
    )
    adapter_class = type(
        definition.original.__name__,
        (definition.original,),
        namespace,
    )
    options = _supported_options(daft.cls, _operator_options(definition), "daft.cls")
    return daft.cls(adapter_class, **options)
