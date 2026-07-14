"""Checks that happen before a model is loaded or changed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DISK_MULTIPLIER = 3


@dataclass(frozen=True)
class PreflightReport:
    ready: bool
    required_disk_bytes: int
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


def build_preflight_report(
    *,
    model_bytes: int,
    available_disk_bytes: int,
    available_ram_bytes: int,
    available_vram_bytes: int | None,
    required_vram_bytes: int,
) -> PreflightReport:
    """Return a conservative, side-effect-free resource assessment."""
    required_disk_bytes = model_bytes * DISK_MULTIPLIER
    blockers: list[str] = []
    warnings: list[str] = []

    if available_disk_bytes < required_disk_bytes:
        blockers.append(
            f"disk: need {required_disk_bytes} bytes, found {available_disk_bytes} bytes"
        )
    if available_ram_bytes < model_bytes:
        blockers.append(
            f"ram: need at least {model_bytes} bytes, found {available_ram_bytes} bytes"
        )
    if required_vram_bytes > 0:
        if available_vram_bytes is None:
            warnings.append("vram: could not detect accelerator memory")
        elif available_vram_bytes < required_vram_bytes:
            blockers.append(
                f"vram: need {required_vram_bytes} bytes, found {available_vram_bytes} bytes"
            )

    return PreflightReport(
        ready=not blockers,
        required_disk_bytes=required_disk_bytes,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def validate_output_directory(workspace: Path, requested_name: str) -> Path | None:
    """Resolve an output name only when it stays inside the selected workspace."""
    if not requested_name or Path(requested_name).is_absolute():
        return None
    workspace_root = workspace.resolve()
    candidate = (workspace_root / requested_name).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        return None
    return candidate
