from __future__ import annotations

import functools
from threading import Lock
from typing import Any, Callable, TypeVar, overload

import pyarrow as pa

from ._definitions import (
    EntrypointDefinition,
    ErrorPolicy,
    OperatorDefinition,
    Resources,
)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T", bound=type)


class FunctionFacade:
    """A lazy callable that retains an mmops entrypoint definition."""

    def __init__(self, definition: EntrypointDefinition) -> None:
        self.definition = definition
        self._compiled: Callable[..., Any] | None = None
        self._compile_lock = Lock()
        functools.update_wrapper(self, definition.original)
        self.__signature__ = __import__("inspect").signature(definition.original)

    def _compile(self) -> Callable[..., Any]:
        if self._compiled is None:
            with self._compile_lock:
                if self._compiled is None:
                    from .backends.daft import compile_function

                    self._compiled = compile_function(self.definition)
        return self._compiled

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._compile()(*args, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_compile_lock", None)
        state["_compiled"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._compile_lock = Lock()


class OperatorFacade:
    """A lazy class facade that compiles an operator for Daft on construction."""

    def __init__(self, definition: OperatorDefinition) -> None:
        self.definition = definition
        self._compiled: type | None = None
        self._compile_lock = Lock()
        functools.update_wrapper(self, definition.original, updated=())
        self.__signature__ = __import__("inspect").signature(definition.original)

    def _compile(self) -> type:
        if self._compiled is None:
            with self._compile_lock:
                if self._compiled is None:
                    from .backends.daft import compile_operator

                    self._compiled = compile_operator(self.definition)
        return self._compiled

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        backend_instance = self._compile()(*args, **kwargs)
        return OperatorInstanceFacade(self.definition, backend_instance)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_compile_lock", None)
        state["_compiled"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._compile_lock = Lock()


class OperatorInstanceFacade:
    """Expose only explicitly declared entrypoints from a backend instance."""

    def __init__(self, definition: OperatorDefinition, backend_instance: Any) -> None:
        self._definition = definition
        self._backend_instance = backend_instance

    def __getattr__(self, name: str) -> Any:
        definition = self.__dict__.get("_definition")
        if definition is None:
            raise AttributeError(name)
        if name not in definition.entrypoints:
            raise AttributeError(
                f"operator {definition.original.__name__!r} has no "
                f"mmops entrypoint {name!r}"
            )
        backend_instance = self.__dict__.get("_backend_instance")
        if backend_instance is None:
            raise AttributeError(name)
        return getattr(backend_instance, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if "__call__" not in self._definition.entrypoints:
            raise TypeError(
                f"operator {self._definition.original.__name__!r} is not callable"
            )
        return self.__getattr__("__call__")(*args, **kwargs)


def _entrypoint(
    kind: str,
    fn: F | None = None,
    *,
    return_dtype: pa.DataType | None = None,
    resources: Resources | None = None,
    unnest: bool = False,
    batch_size: int | None = None,
    max_retries: int | None = None,
    on_error: ErrorPolicy | None = None,
    use_process: bool | None = None,
    backend_options: dict[str, Any] | None = None,
) -> FunctionFacade | Callable[[F], FunctionFacade]:
    def decorate(original: F) -> FunctionFacade:
        return FunctionFacade(
            EntrypointDefinition(
                original=original,
                kind=kind,  # type: ignore[arg-type]
                return_dtype=return_dtype,
                resources=resources or Resources(),
                unnest=unnest,
                batch_size=batch_size,
                max_retries=max_retries,
                on_error=on_error,
                use_process=use_process,
                backend_options=backend_options or {},
            )
        )

    return decorate if fn is None else decorate(fn)


@overload
def row(fn: F) -> FunctionFacade: ...


@overload
def row(
    *,
    return_dtype: pa.DataType | None = None,
    resources: Resources | None = None,
    unnest: bool = False,
    max_retries: int | None = None,
    on_error: ErrorPolicy | None = None,
    use_process: bool | None = None,
    backend_options: dict[str, Any] | None = None,
) -> Callable[[F], FunctionFacade]: ...


def row(
    fn: F | None = None,
    *,
    return_dtype: pa.DataType | None = None,
    resources: Resources | None = None,
    unnest: bool = False,
    max_retries: int | None = None,
    on_error: ErrorPolicy | None = None,
    use_process: bool | None = None,
    backend_options: dict[str, Any] | None = None,
) -> FunctionFacade | Callable[[F], FunctionFacade]:
    """Define a row-wise stateless function or operator method."""

    return _entrypoint(
        "row",
        fn,
        return_dtype=return_dtype,
        resources=resources,
        unnest=unnest,
        max_retries=max_retries,
        on_error=on_error,
        use_process=use_process,
        backend_options=backend_options,
    )


def batch(
    fn: F | None = None,
    *,
    return_dtype: pa.DataType | None = None,
    resources: Resources | None = None,
    unnest: bool = False,
    batch_size: int | None = None,
    max_retries: int | None = None,
    on_error: ErrorPolicy | None = None,
    use_process: bool | None = None,
    backend_options: dict[str, Any] | None = None,
) -> FunctionFacade | Callable[[F], FunctionFacade]:
    """Define an Arrow-array batch function or operator method."""

    return _entrypoint(
        "batch",
        fn,
        return_dtype=return_dtype,
        resources=resources,
        unnest=unnest,
        batch_size=batch_size,
        max_retries=max_retries,
        on_error=on_error,
        use_process=use_process,
        backend_options=backend_options,
    )


@overload
def operator(class_: T) -> OperatorFacade: ...


@overload
def operator(
    *,
    resources: Resources | None = None,
    max_concurrency: int | None = None,
    max_retries: int | None = None,
    on_error: ErrorPolicy | None = None,
    use_process: bool | None = None,
    name: str | None = None,
    backend_options: dict[str, Any] | None = None,
) -> Callable[[T], OperatorFacade]: ...


def operator(
    class_: T | None = None,
    *,
    resources: Resources | None = None,
    max_concurrency: int | None = None,
    max_retries: int | None = None,
    on_error: ErrorPolicy | None = None,
    use_process: bool | None = None,
    name: str | None = None,
    backend_options: dict[str, Any] | None = None,
) -> OperatorFacade | Callable[[T], OperatorFacade]:
    """Define a stateful operator whose marked methods are UDF entrypoints."""

    def decorate(original: T) -> OperatorFacade:
        entrypoints = {
            attr_name: attr.definition
            for attr_name, attr in vars(original).items()
            if isinstance(attr, FunctionFacade)
        }
        return OperatorFacade(
            OperatorDefinition(
                original=original,
                entrypoints=entrypoints,
                resources=resources or Resources(),
                max_concurrency=max_concurrency,
                max_retries=max_retries,
                on_error=on_error,
                use_process=use_process,
                name=name,
                backend_options=backend_options or {},
            )
        )

    return decorate if class_ is None else decorate(class_)
