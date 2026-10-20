from __future__ import annotations

import json
import subprocess
import sys

import pytest

import mmops.operators as operators


def _run_catalog_probe(access: str | None = None) -> dict[str, bool]:
    access_statement = f"operators.{access}\n" if access is not None else ""
    script = f"""
import json
import sys

import mmops.operators as operators

{access_statement}print(json.dumps({{
    "text": "mmops.operators.text" in sys.modules,
    "robotics": "mmops.operators.robotics" in sys.modules,
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_importing_root_catalog_does_not_import_domain_catalogs() -> None:
    assert _run_catalog_probe() == {"text": False, "robotics": False}


def test_accessing_text_export_loads_only_text_catalog() -> None:
    assert _run_catalog_probe("KeywordMatcher") == {
        "text": True,
        "robotics": False,
    }


def test_accessing_robotics_export_loads_only_robotics_catalog() -> None:
    assert _run_catalog_probe("McapVideoExtractor") == {
        "text": False,
        "robotics": True,
    }


def test_unknown_root_catalog_export_raises_standard_attribute_error() -> None:
    with pytest.raises(
        AttributeError,
        match="module 'mmops.operators' has no attribute 'unknown_operator'",
    ):
        operators.unknown_operator
