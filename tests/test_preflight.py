from pathlib import Path

from metaflora_incubus.preflight import build_preflight_report, validate_output_directory


def test_preflight_marks_run_blocked_when_disk_budget_is_too_small(tmp_path: Path) -> None:
    report = build_preflight_report(
        model_bytes=10_000,
        available_disk_bytes=20_000,
        available_ram_bytes=1_000_000,
        available_vram_bytes=None,
        required_vram_bytes=0,
    )

    assert report.ready is False
    assert "disk" in report.blockers[0]


def test_output_directory_rejects_escape_from_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "runs"
    workspace.mkdir()

    assert validate_output_directory(workspace, "incubus-v1") == workspace / "incubus-v1"
    assert validate_output_directory(workspace, "../outside") is None


def test_preflight_reports_missing_vram_as_a_warning() -> None:
    report = build_preflight_report(
        model_bytes=10,
        available_disk_bytes=1_000,
        available_ram_bytes=1_000,
        available_vram_bytes=None,
        required_vram_bytes=20,
    )

    assert report.ready is True
    assert report.warnings == ("vram: could not detect accelerator memory",)
