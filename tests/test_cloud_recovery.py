from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from metaflora_incubus.cloud_training import (
    GIB,
    CheckpointBackend,
    CloudConstraintError,
    CloudExecutionPlan,
    RemoteCheckpointTarget,
    load_cloud_config,
)
from metaflora_incubus.cloud_training_runtime import (
    _authenticated_parent_binding,
    _authenticated_recovery_binding,
    _cuda_cmake_arguments,
    _native_build_jobs,
    _reusable_final_gguf,
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


def test_recovery_ignores_only_stale_training_checkpoints(tmp_path: Path) -> None:
    root = signed_checkpoint(tmp_path)
    stale = root / "preference" / "checkpoint-3" / "trainer_state.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")

    binding = _authenticated_recovery_binding(
        root,
        environment=recovery_environment(),
        checkpoint_key="k" * 32,
        run_id="incubus-v1-run",
    )

    assert binding["run_id"] == "incubus-v1-run"


def test_recovery_rejects_fake_training_checkpoint_prefix(tmp_path: Path) -> None:
    root = signed_checkpoint(tmp_path)
    fake = root / "preference" / "checkpoint-evil" / "payload.bin"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"unsigned")

    with pytest.raises(CloudConstraintError, match="integrity verification"):
        _authenticated_recovery_binding(
            root,
            environment=recovery_environment(),
            checkpoint_key="k" * 32,
            run_id="incubus-v1-run",
        )


def test_recovery_allows_pruned_signed_training_checkpoints(tmp_path: Path) -> None:
    root = signed_checkpoint(tmp_path)
    prunable = root / "sft" / "checkpoint-8" / "trainer_state.json"
    prunable.parent.mkdir(parents=True)
    prunable.write_text("{}", encoding="utf-8")
    _write_checkpoint_manifest(root, binding=checkpoint_binding(), key="k" * 32)
    prunable.unlink()
    prunable.parent.rmdir()
    prunable.parent.parent.rmdir()

    binding = _authenticated_recovery_binding(
        root,
        environment=recovery_environment(),
        checkpoint_key="k" * 32,
        run_id="incubus-v1-run",
    )

    assert binding["run_id"] == "incubus-v1-run"


def test_recovery_allows_pruned_signed_training_stage_outputs(tmp_path: Path) -> None:
    root = signed_checkpoint(tmp_path)
    stage_readme = root / "sft" / "README.md"
    stage_adapter = root / "sft" / "final" / "adapter_model.safetensors"
    stage_adapter.parent.mkdir(parents=True)
    stage_readme.write_text("temporary stage output", encoding="utf-8")
    stage_adapter.write_bytes(b"temporary adapter")
    _write_checkpoint_manifest(root, binding=checkpoint_binding(), key="k" * 32)
    stage_adapter.unlink()
    stage_adapter.parent.rmdir()
    stage_readme.unlink()
    stage_readme.parent.rmdir()

    binding = _authenticated_recovery_binding(
        root,
        environment=recovery_environment(),
        checkpoint_key="k" * 32,
        run_id="incubus-v1-run",
    )

    assert binding["run_id"] == "incubus-v1-run"


def test_recovery_rejects_untracked_final_adapter_file(tmp_path: Path) -> None:
    root = signed_checkpoint(tmp_path)
    (root / "final-adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CloudConstraintError, match="integrity verification"):
        _authenticated_recovery_binding(
            root,
            environment=recovery_environment(),
            checkpoint_key="k" * 32,
            run_id="incubus-v1-run",
        )


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


def test_refinement_accepts_authenticated_parent_with_an_older_dataset(tmp_path: Path) -> None:
    root = signed_checkpoint(tmp_path)
    environment = {
        **recovery_environment(),
        "INCUBUS_DATASET_REVISION": "9" * 40,
        "INCUBUS_DATASET_SHA256": "8" * 64,
    }

    binding = _authenticated_parent_binding(
        root,
        environment=environment,
        checkpoint_key="k" * 32,
        parent_run_id="incubus-v1-run",
    )

    assert binding["dataset_revision"] == "2" * 40
    assert (root / "final-adapter" / "adapter_model.safetensors").is_file()


def test_refinement_verifies_adapter_only_subset_of_parent_with_artifacts(tmp_path: Path) -> None:
    full = signed_checkpoint(tmp_path / "full")
    (full / "final-adapter" / "adapter_config.json").write_text("{}\n")
    artifacts = full / "artifacts"
    artifacts.mkdir()
    (artifacts / "model.gguf").write_bytes(b"large release artifact")
    _write_checkpoint_manifest(full, binding=checkpoint_binding(), key="k" * 32)
    subset = tmp_path / "subset"
    shutil.copytree(full / "final-adapter", subset / "final-adapter")
    shutil.copy2(full / "incubus-checkpoint-manifest.json", subset)

    binding = _authenticated_parent_binding(
        subset,
        environment=recovery_environment(),
        checkpoint_key="k" * 32,
        parent_run_id="incubus-v1-run",
    )

    assert binding["run_id"] == "incubus-v1-run"

    (subset / "final-adapter" / "adapter_config.json").unlink()
    with pytest.raises(CloudConstraintError, match="incomplete"):
        _authenticated_parent_binding(
            subset,
            environment=recovery_environment(),
            checkpoint_key="k" * 32,
            parent_run_id="incubus-v1-run",
        )


def test_refinement_rejects_parent_source_drift(tmp_path: Path) -> None:
    root = signed_checkpoint(tmp_path)
    environment = {**recovery_environment(), "INCUBUS_SOURCE_REVISION": "7" * 40}

    with pytest.raises(CloudConstraintError, match="parent source revision"):
        _authenticated_parent_binding(
            root,
            environment=environment,
            checkpoint_key="k" * 32,
            parent_run_id="incubus-v1-run",
        )


def test_cuda_cmake_arguments_use_kaggle_compat_driver(tmp_path: Path) -> None:
    driver = tmp_path / "compat" / "libcuda.so"
    driver.parent.mkdir()
    driver.write_bytes(b"driver")

    arguments = _cuda_cmake_arguments(
        cuda_enabled=True,
        cuda_driver_path=driver,
    )

    assert "-DGGML_CUDA=ON" in arguments
    assert f"-DCMAKE_LIBRARY_PATH={driver.parent}" in arguments
    assert _cuda_cmake_arguments(cuda_enabled=False) == ["-DGGML_CUDA=OFF"]


@pytest.mark.parametrize(
    ("cpu_count", "expected"),
    ((None, 1), (0, 1), (1, 1), (4, 4), (64, 4)),
)
def test_native_build_jobs_are_bounded(cpu_count: int | None, expected: int) -> None:
    assert _native_build_jobs(cpu_count=cpu_count) == expected


def recovery_plan(tmp_path: Path) -> CloudExecutionPlan:
    config = load_cloud_config(Path("configs/cloud/free-gpu-v1.json"))
    target = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location="metaflora/incubus-checkpoints",
        branch="incubus-training-v1",
    )
    plan = CloudExecutionPlan.create(
        config=config,
        checkpoint_target=target,
        run_id="incubus-v1-run",
        parameter_count=4_659_865_088,
        vram_bytes=16 * GIB,
    )
    return plan


def test_refinement_plan_is_dpo_only_and_keeps_parent_immutable() -> None:
    config = load_cloud_config(Path("configs/cloud/free-gpu-v1.json"))
    target = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location="metaflora/incubus-checkpoints",
        branch="incubus-training-v1",
    )

    plan = CloudExecutionPlan.create_refinement(
        config=config,
        checkpoint_target=target,
        run_id="incubus-v1-refine-001",
        parent_run_id="incubus-v1-run",
        parameter_count=4_659_865_088,
        vram_bytes=16 * GIB,
    )

    assert plan.training_mode == "dpo_refinement"
    assert plan.parent_run_id == "incubus-v1-run"
    assert "sft" not in plan.post_training_steps
    assert plan.workspace.name == "incubus-v1-refine-001"


def test_cpu_recovery_plan_validates_parameter_limit_without_weakening_gpu_plan() -> None:
    config = load_cloud_config(Path("configs/cloud/free-gpu-v1.json"))
    target = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location="metaflora/incubus-checkpoints",
        branch="incubus-training-v1",
    )

    plan = CloudExecutionPlan.create_cpu_recovery(
        config=config,
        checkpoint_target=target,
        run_id="incubus-v1-cpu",
        parameter_count=4_659_865_088,
    )

    assert plan.training_mode == "recovery_only_cpu"
    with pytest.raises(CloudConstraintError, match="parameter"):
        CloudExecutionPlan.create_cpu_recovery(
            config=config,
            checkpoint_target=target,
            run_id="incubus-v1-too-large",
            parameter_count=config.profile.max_parameter_count + 1,
        )
    with pytest.raises(CloudConstraintError, match="VRAM"):
        CloudExecutionPlan.create(
            config=config,
            checkpoint_target=target,
            run_id="incubus-v1-gpu",
            parameter_count=4_659_865_088,
            vram_bytes=0,
        )


def test_recovery_reuses_valid_private_gguf(tmp_path: Path) -> None:
    plan = recovery_plan(tmp_path)
    artifact = tmp_path / "checkpoints" / "artifacts" / "metaflora-incubus-v1.gguf"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    with artifact.open("r+b") as handle:
        handle.truncate(5 * GIB // 2)
    server = artifact.parent / "llama-server"
    server.write_bytes(b"static-server")

    assert _reusable_final_gguf(plan=plan, checkpoint_root=tmp_path / "checkpoints") == artifact


def test_recovery_rejects_private_gguf_without_synced_server(tmp_path: Path) -> None:
    plan = recovery_plan(tmp_path)
    artifact = tmp_path / "checkpoints" / "artifacts" / "metaflora-incubus-v1.gguf"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    with artifact.open("r+b") as handle:
        handle.truncate(5 * GIB // 2)

    assert _reusable_final_gguf(plan=plan, checkpoint_root=tmp_path / "checkpoints") is None


@pytest.mark.parametrize("size_bytes", (0, 5 * GIB // 2 - 1, 5 * GIB + 1))
def test_recovery_rejects_incomplete_private_gguf(
    tmp_path: Path, size_bytes: int
) -> None:
    plan = recovery_plan(tmp_path)
    artifact = tmp_path / "checkpoints" / "artifacts" / "metaflora-incubus-v1.gguf"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    with artifact.open("r+b") as handle:
        handle.truncate(size_bytes)

    assert _reusable_final_gguf(plan=plan, checkpoint_root=tmp_path / "checkpoints") is None
