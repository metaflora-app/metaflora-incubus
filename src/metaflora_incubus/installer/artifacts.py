"""Select immutable release artifacts that fit a target machine."""

from __future__ import annotations

from dataclasses import dataclass


class NoCompatibleArtifactError(RuntimeError):
    """Raised when no published artifact fits the target machine."""


@dataclass(frozen=True)
class ReleaseArtifact:
    artifact_id: str
    platform: str
    architecture: str
    download_url: str
    sha256: str
    download_bytes: int
    required_ram_bytes: int
    required_disk_bytes: int
    priority: int


def select_release_artifact(
    artifacts: tuple[ReleaseArtifact, ...],
    *,
    platform: str,
    architecture: str,
    available_ram_bytes: int,
    available_disk_bytes: int,
) -> ReleaseArtifact:
    compatible = tuple(
        artifact
        for artifact in artifacts
        if artifact.platform == platform
        and artifact.architecture == architecture
        and artifact.required_ram_bytes <= available_ram_bytes
        and artifact.required_disk_bytes <= available_disk_bytes
    )
    if not compatible:
        raise NoCompatibleArtifactError(
            f"no artifact for {platform}/{architecture} with available resources"
        )
    return max(compatible, key=lambda artifact: (artifact.priority, artifact.artifact_id))
