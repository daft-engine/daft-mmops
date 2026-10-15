from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _extras() -> dict[str, list[str]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return project["optional-dependencies"]


def _names(requirements: list[str]) -> set[str]:
    return {
        requirement.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0]
        for requirement in requirements
    }


def test_text_extra_installs_the_current_daft_runtime() -> None:
    extras = _extras()

    assert "text" in extras
    assert "daft" in _names(extras["text"])


def test_all_extra_contains_every_released_domain_dependency() -> None:
    extras = _extras()
    domains = (
        "text",
        "image",
        "audio",
        "video",
        "robotics",
        "document",
        "multimodal",
    )
    domain_dependencies = {
        requirement for domain in domains for requirement in extras[domain]
    }

    assert "all" in extras
    assert set(extras["all"]) == domain_dependencies


def test_robotics_extra_installs_mcap_remux_and_tos_filesystem_dependencies() -> None:
    extras = _extras()

    assert {"daft", "mcap", "mcap-ros2-support", "av", "fsspec", "tosfs"} <= _names(
        extras["robotics"]
    )
    assert "tos" not in _names(extras["robotics"])


def test_mcap_video_guide_is_linked_and_documents_worker_configuration() -> None:
    guide_path = ROOT / "docs" / "operators" / "robotics" / "mcap-video-extractor.md"

    assert (
        "(docs/operators/robotics/mcap-video-extractor.md)"
        in (ROOT / "README.md").read_text()
    )
    guide = guide_path.read_text()
    for variable in (
        "TOS_ENDPOINT",
        "TOS_REGION",
        "TOS_ACCESS_KEY",
        "TOS_SECRET_KEY",
        "TOS_SECURITY_TOKEN",
    ):
        assert f"`{variable}`" in guide


def test_test_extra_installs_dependencies_imported_by_the_full_suite() -> None:
    extras = _extras()

    assert {
        "daft",
        "mcap",
        "mcap-ros2-support",
        "av",
        "fsspec",
        "tosfs",
        "pytest",
    } <= _names(extras["test"])


def test_unreleased_domain_extras_are_reserved_and_empty() -> None:
    extras = _extras()

    for domain in ("image", "audio", "video", "document", "multimodal"):
        assert domain in extras
        assert extras[domain] == []


def test_dev_extra_supports_every_documented_verification_command() -> None:
    extras = _extras()

    assert {
        "daft",
        "mcap",
        "mcap-ros2-support",
        "av",
        "fsspec",
        "tosfs",
        "pytest",
        "black",
        "build",
        "twine",
    } <= _names(extras["dev"])
