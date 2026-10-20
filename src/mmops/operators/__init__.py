"""Multimodal data processing operators."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .robotics import MCAP_VIDEO_RESULT_DTYPE, McapVideoExtractor
    from .text import KEYWORD_MATCH_DTYPE, KeywordMatcher, normalize_text

__all__ = [
    "KEYWORD_MATCH_DTYPE",
    "MCAP_VIDEO_RESULT_DTYPE",
    "KeywordMatcher",
    "McapVideoExtractor",
    "normalize_text",
]

_EXPORT_MODULES = {
    "KEYWORD_MATCH_DTYPE": ".text",
    "KeywordMatcher": ".text",
    "normalize_text": ".text",
    "MCAP_VIDEO_RESULT_DTYPE": ".robotics",
    "McapVideoExtractor": ".robotics",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
