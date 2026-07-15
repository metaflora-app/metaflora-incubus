from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from metaflora_incubus.cloud_training import CloudConstraintError
from metaflora_incubus.cloud_training_runtime import (
    _authenticated_recovery_binding,
    _write_checkpoint_manifest,
)


def recovery_environment() -> dict[str, str]:
    return {
        "INCUBUS_SOURCE_REPO": "private/source",
        "INCUBUS_SOURCE_REVISION": "1" * 40,
        "INCUBUS_DATASET_REPO": "private/data",
        "INCUBUS_DATASET_REVISION": "2" * 40,
        "INCUBUS_DATASET_SHA256": "3" * 64,
    }


def checkpoint_binding() -> dict[str, str]:
    environment = recovery_environment()
    return {
        "code_revision": "0" * 40,
        "config_sha256": "4" * 64,
        "dataset_repo_sha256": hashlib.sha256(
            environment["INCUBUS_DATASET_REPO"].encode()
        ).hexdigest(),
        "dataset_revision": environment["INCUBUS_DATASET_REVISION"],
        "dataset_sha256": environment["INCUBUS_DATASET_SHA256"],
        "run_id": "incubus-v1-run",
        "source_repo_sha256": hashlib.sha256(
            environment["INCUBUS_SOURCE_REPO"].encode()
        ).hexdigest(),
        "source_revision": environment["INCUBUS_SOURCE_REVISION"],
    }


def signed_checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "checkpoints"
    adapter = root / "final-adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"trained")
    _write_checkpoint_manifest(root, binding=checkpoint_binding(), key="k" * 32)
    return root


def test_recovery_accepts_authenticated_checkpoint_from_older_code_revision(
    tmp_path: Path,
) -> None:
    root = signed_checkpoint(tmp_path)

    binding = _authenticated_recovery_binding(
        root,
        environment=recovery_environment(),
        checkpoint_key="k" * 32,
        run_id="incubus-v1-run",
    )

    assert binding["code_revision"] == "0" * 40


def test_recovery_rejects_source_identity_drift(tmp_path: Path) -> None:
    root = signed_checkpoint(tmp_path)
    environment = {**recovery_environment(), "INCUBUS_SOURCE_REPO": "wrong/source"}

    with pytest.raises(CloudConstraintError, match="source identity"):
        _authenticated_recovery_binding(
            root,
            environment=environment,
            checkpoint_key="k" * 32,
            run_id="incubus-v1-run",
        )
