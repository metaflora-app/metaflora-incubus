from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from metaflora_incubus.cloud_training import CloudConfig, FreeGpuProfile


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "incubus_test_recover_free_gpu", Path("scripts/recover_free_gpu.py")
    )
    assert specification is not None and specification.loader is not None
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    return runner


def test_cpu_fallback_never_probes_nvidia_and_propagates_explicit_mode(monkeypatch) -> None:
    runner = load_runner()
    plan = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "load_cloud_config", lambda path: object())
    monkeypatch.setattr(runner, "detect_code_revision", lambda: "a" * 40)
    monkeypatch.setattr(
        runner,
        "detect_vram_bytes",
        lambda: (_ for _ in ()).throw(AssertionError("nvidia-smi must not run")),
    )
    monkeypatch.setattr(
        runner.CloudExecutionPlan,
        "create_cpu_recovery",
        lambda **values: plan,
    )
    monkeypatch.setattr(
        runner,
        "recover_trained_artifact",
        lambda **values: (
            captured.update({**values, "environment": dict(values["environment"])})
            or {"case_count": 48}
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_with_failure_reporting",
        lambda operation, **values: operation(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover_free_gpu.py",
            "--cpu-fallback",
            "--run-id",
            "incubus-v1-run",
            "--parameter-count",
            "4659865088",
            "--checkpoint-location",
            "metaflora/incubus-checkpoints",
            "--checkpoint-branch",
            "incubus-training-v1",
        ],
    )

    assert runner.main() == 0
    assert captured["plan"] is plan
    assert captured["cpu_fallback"] is True
    assert captured["environment"]["INCUBUS_CODE_REVISION"] == "a" * 40


def test_workspace_root_overrides_colab_default_for_kaggle(monkeypatch) -> None:
    runner = load_runner()
    captured: dict[str, object] = {}
    plan = object()
    monkeypatch.setattr(runner, "detect_code_revision", lambda: "a" * 40)
    monkeypatch.setattr(runner, "detect_vram_bytes", lambda: 32 * 1024**3)
    monkeypatch.setattr(
        runner,
        "load_cloud_config",
        lambda path: CloudConfig(
            product_id="metaflora-incubus-v1",
            workspace=Path("/content/incubus-work"),
            public_repo_id="metaflora/incubus",
            profile=FreeGpuProfile.default(),
            llama_cpp_revision="1" * 40,
            config_sha256="2" * 64,
        ),
    )
    monkeypatch.setattr(
        runner.CloudExecutionPlan,
        "create",
        lambda **values: captured.update(values) or plan,
    )
    monkeypatch.setattr(
        runner,
        "recover_trained_artifact",
        lambda **values: {"case_count": 48},
    )
    monkeypatch.setattr(
        runner,
        "run_with_failure_reporting",
        lambda operation, **values: operation(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover_free_gpu.py",
            "--run-id",
            "incubus-v1-run",
            "--parameter-count",
            "4659865088",
            "--checkpoint-location",
            "metaflora/incubus-checkpoints",
            "--checkpoint-branch",
            "incubus-training-v1",
            "--workspace-root",
            "/kaggle/working/incubus-work",
        ],
    )

    assert runner.main() == 0
    assert captured["config"].workspace == Path("/kaggle/working/incubus-work")
