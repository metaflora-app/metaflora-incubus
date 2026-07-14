from __future__ import annotations

import importlib

import pytest

GIB = 1024**3


def _artifacts_module():
    return importlib.import_module("metaflora_incubus.installer.artifacts")


def _release_artifacts():
    artifacts = _artifacts_module()
    return (
        artifacts.ReleaseArtifact(
            artifact_id="incubus-v1-compact",
            platform="darwin",
            architecture="arm64",
            download_url="https://releases.metaflora.ai/incubus/v1/compact.tar.zst",
            sha256="1" * 64,
            download_bytes=3 * GIB,
            required_ram_bytes=8 * GIB,
            required_disk_bytes=8 * GIB,
            priority=10,
        ),
        artifacts.ReleaseArtifact(
            artifact_id="incubus-v1-performance",
            platform="darwin",
            architecture="arm64",
            download_url="https://releases.metaflora.ai/incubus/v1/performance.tar.zst",
            sha256="2" * 64,
            download_bytes=5 * GIB,
            required_ram_bytes=16 * GIB,
            required_disk_bytes=12 * GIB,
            priority=20,
        ),
        artifacts.ReleaseArtifact(
            artifact_id="incubus-v1-linux",
            platform="linux",
            architecture="x86_64",
            download_url="https://releases.metaflora.ai/incubus/v1/linux.tar.zst",
            sha256="3" * 64,
            download_bytes=5 * GIB,
            required_ram_bytes=16 * GIB,
            required_disk_bytes=12 * GIB,
            priority=20,
        ),
    )


def test_selects_highest_priority_artifact_that_fits_the_machine() -> None:
    artifacts = _artifacts_module()

    selected = artifacts.select_release_artifact(
        _release_artifacts(),
        platform="darwin",
        architecture="arm64",
        available_ram_bytes=24 * GIB,
        available_disk_bytes=20 * GIB,
    )

    assert selected.artifact_id == "incubus-v1-performance"


def test_falls_back_to_compact_artifact_when_ram_is_limited() -> None:
    artifacts = _artifacts_module()

    selected = artifacts.select_release_artifact(
        _release_artifacts(),
        platform="darwin",
        architecture="arm64",
        available_ram_bytes=12 * GIB,
        available_disk_bytes=20 * GIB,
    )

    assert selected.artifact_id == "incubus-v1-compact"


@pytest.mark.parametrize(
    ("platform", "architecture", "ram_gib", "disk_gib"),
    [
        ("win32", "x86_64", 32, 32),
        ("darwin", "x86_64", 32, 32),
        ("darwin", "arm64", 7, 32),
        ("darwin", "arm64", 32, 7),
    ],
)
def test_rejects_machines_without_a_compatible_artifact(
    platform: str,
    architecture: str,
    ram_gib: int,
    disk_gib: int,
) -> None:
    artifacts = _artifacts_module()

    with pytest.raises(artifacts.NoCompatibleArtifactError):
        artifacts.select_release_artifact(
            _release_artifacts(),
            platform=platform,
            architecture=architecture,
            available_ram_bytes=ram_gib * GIB,
            available_disk_bytes=disk_gib * GIB,
        )
