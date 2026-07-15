from __future__ import annotations

from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _project_requirements() -> dict[str, Requirement]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = (Requirement(value) for value in project["dependencies"])
    return {requirement.name: requirement for requirement in requirements}


def _input_requirements(filename: str) -> dict[str, Requirement]:
    values = (
        Requirement(line)
        for line in (ROOT / "requirements" / filename).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    return {requirement.name: requirement for requirement in values}


def test_project_huggingface_constraint_matches_transformers_v5_and_locked_cloud_runtime() -> None:
    requirements = _project_requirements()
    hub_specifier = requirements["huggingface-hub"].specifier
    transformers_specifier = requirements["transformers"].specifier

    # Transformers 5.x requires huggingface-hub >=1.5,<2, while both reproducible
    # cloud lock inputs pin 1.23.0. The package metadata must admit that exact pin.
    assert Version("1.5.0") in hub_specifier
    assert Version("2.0.0") not in hub_specifier

    for filename in ("cloud.in", "recovery.in"):
        locked = _input_requirements(filename)
        assert str(locked["huggingface-hub"].specifier) == "==1.23.0"
        assert Version("1.23.0") in hub_specifier
        assert str(locked["transformers"].specifier) == "==5.13.1"
        assert Version("5.13.1") in transformers_specifier


def test_cryptography_constraint_and_cloud_inputs_exclude_vulnerable_wheels() -> None:
    requirements = _project_requirements()
    cryptography_specifier = requirements["cryptography"].specifier

    assert Version("48.0.1") in cryptography_specifier
    assert Version("48.0.0") not in cryptography_specifier
    assert Version("49.0.0") not in cryptography_specifier
    for filename in ("cloud.in", "recovery.in", "h2h.in"):
        locked = _input_requirements(filename)
        assert str(locked["cryptography"].specifier) == "==48.0.1"
