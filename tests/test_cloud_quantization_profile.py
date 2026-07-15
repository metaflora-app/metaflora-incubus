from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from metaflora_incubus.cloud_training import (
    GIB,
    CheckpointBackend,
    CloudExecutionPlan,
    RemoteCheckpointTarget,
    load_cloud_config,
)
from metaflora_incubus.cloud_training_runtime import _build_gguf

CONFIG_PATH = Path("configs/cloud/free-gpu-v1.json")


def cloud_plan(tmp_path: Path) -> CloudExecutionPlan:
    config = load_cloud_config(CONFIG_PATH)
    target = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location="metaflora/incubus-checkpoints",
        branch="incubus-training-v1",
    )
    plan = CloudExecutionPlan.create(
        config=config,
        checkpoint_target=target,
        run_id="q5-release-test",
        parameter_count=4_660_000_000,
        vram_bytes=16 * GIB,
    )
    return replace(plan, workspace=tmp_path / "workspace")


def test_compact_cloud_profile_declares_q5_in_config_and_plan(tmp_path: Path) -> None:
    plan = cloud_plan(tmp_path)

    assert plan.config.profile.final_gguf_quantization == "Q5_K_M"
    assert plan.final_gguf_quantization == "Q5_K_M"
    assert "quantize_q5_k_m" in plan.post_training_steps
    assert "quantize_q4_k_m" not in plan.post_training_steps
    assert "run_candidate_benchmark" in plan.post_training_steps
    assert "sync_private_evidence" in plan.post_training_steps


def test_cloud_export_invokes_q5_quantizer_and_preserves_release_filename(
    monkeypatch, tmp_path: Path
) -> None:
    plan = cloud_plan(tmp_path)
    source = tmp_path / "source"
    adapter = tmp_path / "adapter"
    artifacts = tmp_path / "artifacts"
    source.mkdir()
    adapter.mkdir()
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd=None) -> None:
        commands.append(command)
        if command[:2] == ["cmake", "--build"]:
            server = Path(cwd) / "build/bin/llama-server"
            server.parent.mkdir(parents=True, exist_ok=True)
            server.write_bytes(b"server")
        if command[-1] == "Q5_K_M":
            output = Path(command[-2])
            output.touch()
            with output.open("r+b") as handle:
                handle.truncate(5 * GIB // 2)

    monkeypatch.setattr(
        "metaflora_incubus.cloud_training_runtime._checkout_pinned_revision",
        lambda repository, revision: None,
    )
    monkeypatch.setattr("metaflora_incubus.cloud_training_runtime._run", fake_run)

    final = _build_gguf(plan=plan, source=source, adapter=adapter, artifacts=artifacts)

    assert final.name == "metaflora-incubus-v1.gguf"
    build_command = next(command for command in commands if command[:2] == ["cmake", "--build"])
    assert build_command[:-1] == [
        "cmake",
        "--build",
        "build",
        "--config",
        "Release",
        "--target",
        "llama-server",
        "llama-export-lora",
        "llama-quantize",
    ]
    assert build_command[-1] in {"-j1", "-j2", "-j3", "-j4"}
    assert [
        "cmake",
        "-B",
        "build",
        "-DLLAMA_CURL=OFF",
        "-DGGML_CUDA=ON",
        "-DBUILD_SHARED_LIBS=OFF",
    ] in commands
    assert any(command[-1] == "Q5_K_M" for command in commands)
    assert not any(command[-1] == "Q4_K_M" for command in commands)


def test_cpu_recovery_build_disables_cuda_without_changing_release_export(
    monkeypatch, tmp_path: Path
) -> None:
    plan = cloud_plan(tmp_path)
    source = tmp_path / "source-cpu"
    adapter = tmp_path / "adapter-cpu"
    artifacts = tmp_path / "artifacts-cpu"
    source.mkdir()
    adapter.mkdir()
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd=None) -> None:
        commands.append(command)
        if command[:2] == ["cmake", "--build"]:
            server = Path(cwd) / "build/bin/llama-server"
            server.parent.mkdir(parents=True, exist_ok=True)
            server.write_bytes(b"cpu-server")
        if command[-1] == "Q5_K_M":
            output = Path(command[-2])
            output.touch()
            with output.open("r+b") as handle:
                handle.truncate(5 * GIB // 2)

    monkeypatch.setattr(
        "metaflora_incubus.cloud_training_runtime._checkout_pinned_revision",
        lambda repository, revision: None,
    )
    monkeypatch.setattr("metaflora_incubus.cloud_training_runtime._run", fake_run)

    final = _build_gguf(
        plan=plan,
        source=source,
        adapter=adapter,
        artifacts=artifacts,
        cuda_enabled=False,
    )

    assert final.name == "metaflora-incubus-v1.gguf"
    assert any("-DGGML_CUDA=OFF" in command for command in commands)
    assert not any("-DGGML_CUDA=ON" in command for command in commands)
