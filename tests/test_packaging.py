from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _extras() -> dict[str, list[str]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return project["optional-dependencies"]


def _names(requirements: list[str]) -> set[str]:
    return {
        requirement.split(";", 1)[0]
        .split("[", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        for requirement in requirements
    }


def test_text_extra_installs_the_current_daft_runtime() -> None:
    extras = _extras()

    assert "text" in extras
    assert "daft" in _names(extras["text"])


def test_all_extra_contains_every_released_modality_dependency() -> None:
    extras = _extras()
    modalities = ("text", "image", "audio", "video", "document", "multimodal")
    modality_dependencies = {
        requirement for modality in modalities for requirement in extras[modality]
    }

    assert "all" in extras
    assert set(extras["all"]) == modality_dependencies


def test_unreleased_modality_extras_are_reserved_and_empty() -> None:
    extras = _extras()

    for modality in ("image", "audio", "video", "document", "multimodal"):
        assert modality in extras
        assert extras[modality] == []


def test_dev_extra_supports_every_documented_verification_command() -> None:
    extras = _extras()

    assert {"daft", "pytest", "black", "build", "twine"} <= _names(extras["dev"])
