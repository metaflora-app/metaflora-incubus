from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from metaflora_incubus.kaggle_recovery_export import (
    GIB,
    ExportRecoveryError,
    PrivateCandidateUploadResult,
    RecoveryExportConfig,
    ResourceSnapshot,
    execute_recovery_export,
    upload_private_candidate,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ample_resources(_path: Path) -> ResourceSnapshot:
    return ResourceSnapshot(100 * GIB, 100 * GIB)


def _config(tmp_path: Path) -> RecoveryExportConfig:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    tools = tmp_path / "tools"
    base.mkdir()
    adapter.mkdir()
    tools.mkdir()
    (base / "model.safetensors").write_bytes(b"base-weights")
    (base / "config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    converter = tools / "convert_hf_to_gguf.py"
    quantizer = tools / "llama-quantize"
    merger = tools / "merge.py"
    converter.write_text("# converter", encoding="utf-8")
    quantizer.write_bytes(b"quantizer")
    merger.write_text("# merger", encoding="utf-8")
    return RecoveryExportConfig.create(
        run_id="incubus-v1-refine-001",
        base_model=base,
        adapter=adapter,
        workspace=tmp_path / "workspace",
        merge_script=merger,
        convert_script=converter,
        quantize_binary=quantizer,
        minimum_ram_bytes=8 * GIB,
        minimum_free_disk_bytes=12 * GIB,
        minimum_output_bytes=8,
        maximum_output_bytes=1024,
    )


def _staged_config(tmp_path: Path) -> RecoveryExportConfig:
    inputs = tmp_path / "recovery-inputs"
    base = inputs / "source"
    adapter = inputs / "checkpoint" / "final-adapter"
    tools = tmp_path / "tools"
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    tools.mkdir()
    (base / "model.safetensors").write_bytes(b"base-weights")
    (base / "config.json").write_text("{}", encoding="utf-8")
    (inputs / ".source.incubus-disposable").write_text(str(base.resolve()), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    converter = tools / "convert_hf_to_gguf.py"
    quantizer = tools / "llama-quantize"
    merger = tools / "merge.py"
    converter.write_text("# converter", encoding="utf-8")
    quantizer.write_bytes(b"quantizer")
    merger.write_text("# merger", encoding="utf-8")
    return RecoveryExportConfig.create(
        run_id="incubus-v1-refine-001",
        base_model=base,
        adapter=adapter,
        workspace=tmp_path / "workspace",
        merge_script=merger,
        convert_script=converter,
        quantize_binary=quantizer,
        minimum_ram_bytes=8 * GIB,
        minimum_free_disk_bytes=24 * GIB,
        minimum_output_bytes=8,
        maximum_output_bytes=1024,
        reclaim_base_after_merge=True,
        reclaim_intermediates=True,
    )


class FakeCommands:
    def __init__(self, *, crash_quantize_once: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.crash_quantize_once = crash_quantize_once

    def __call__(self, command: tuple[str, ...]) -> None:
        self.calls.append(command)
        if "--output" in command:
            destination = Path(command[command.index("--output") + 1])
            destination.mkdir(parents=True)
            (destination / "model.safetensors").write_bytes(b"merged")
            (destination / "config.json").write_text("{}", encoding="utf-8")
            return
        if "--outfile" in command:
            destination = Path(command[command.index("--outfile") + 1])
            destination.write_bytes(b"GGUF-f16-model")
            return
        destination = Path(command[2])
        if command[-1] == "Q5_K_M":
            if self.crash_quantize_once:
                self.crash_quantize_once = False
                destination.write_bytes(b"partial")
                raise RuntimeError("simulated Kaggle disconnect")
            destination.write_bytes(b"GGUF-q5-model")
            return


def test_recovery_preflight_fails_closed_for_ram_and_disk(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(ExportRecoveryError, match="RAM"):
        execute_recovery_export(
            config,
            resources=lambda _path: ResourceSnapshot(
                available_ram_bytes=config.minimum_ram_bytes - 1,
                free_disk_bytes=100 * GIB,
            ),
            command_runner=FakeCommands(),
        )
    with pytest.raises(ExportRecoveryError, match="disk"):
        execute_recovery_export(
            config,
            resources=lambda _path: ResourceSnapshot(
                available_ram_bytes=100 * GIB,
                free_disk_bytes=config.minimum_free_disk_bytes - 1,
            ),
            command_runner=FakeCommands(),
        )


def test_export_is_q5_hashed_and_idempotent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    commands = FakeCommands()

    first = execute_recovery_export(config, resources=_ample_resources, command_runner=commands)
    first_calls = tuple(commands.calls)
    second = execute_recovery_export(config, resources=_ample_resources, command_runner=commands)

    assert len(first_calls) == 3
    assert first_calls[-1][-1] == "Q5_K_M"
    assert commands.calls == list(first_calls)
    assert first.artifact == second.artifact
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.manifest == second.manifest
    assert first.resumed is False
    assert second.resumed is True
    assert first.artifact_sha256 == _sha(first.artifact)
    assert first.artifact.read_bytes().startswith(b"GGUF")
    manifest = json.loads(first.manifest.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "incubus-v1-refine-001"
    assert manifest["gguf_quantization"] == "Q5_K_M"
    assert manifest["artifact"]["sha256"] == first.artifact_sha256
    assert manifest["resume"]["completed_stages"] == ["merge", "convert", "quantize"]


def test_resume_after_quantizer_crash_does_not_repeat_merge_or_convert(tmp_path: Path) -> None:
    config = _config(tmp_path)
    commands = FakeCommands(crash_quantize_once=True)

    with pytest.raises(RuntimeError, match="disconnect"):
        execute_recovery_export(config, resources=_ample_resources, command_runner=commands)
    calls_before_resume = tuple(commands.calls)
    assert len(calls_before_resume) == 3

    result = execute_recovery_export(config, resources=_ample_resources, command_runner=commands)

    assert len(commands.calls) == 4
    assert commands.calls[-1][-1] == "Q5_K_M"
    assert result.artifact.read_bytes() == b"GGUF-q5-model"
    assert not (config.workspace / "metaflora-incubus-v1.gguf.partial").exists()


def test_disk_budget_is_stage_specific_after_initial_preflight(tmp_path: Path) -> None:
    config = _config(tmp_path)
    snapshots = iter(
        (
            ResourceSnapshot(100 * GIB, 13 * GIB),
            ResourceSnapshot(100 * GIB, 2 * GIB),
            ResourceSnapshot(100 * GIB, 2 * GIB),
        )
    )

    result = execute_recovery_export(
        config,
        resources=lambda _path: next(snapshots),
        command_runner=FakeCommands(),
    )

    assert result.artifact.read_bytes() == b"GGUF-q5-model"


def test_staged_export_reclaims_each_verified_predecessor(tmp_path: Path) -> None:
    config = _staged_config(tmp_path)

    class StageAwareCommands(FakeCommands):
        def __call__(self, command: tuple[str, ...]) -> None:
            if "--outfile" in command:
                assert not config.base_model.exists()
                assert (config.workspace / "merged-safetensors").is_dir()
            elif command[-1] == "Q5_K_M":
                assert not (config.workspace / "merged-safetensors").exists()
                assert (config.workspace / "incubus-v1-f16.gguf").is_file()
            super().__call__(command)

    commands = StageAwareCommands()
    result = execute_recovery_export(
        config,
        resources=lambda _path: ResourceSnapshot(100 * GIB, 2 * GIB),
        command_runner=commands,
    )

    assert result.artifact.read_bytes() == b"GGUF-q5-model"
    assert not config.base_model.exists()
    assert config.adapter.is_dir()
    assert not (config.workspace / "merged-safetensors").exists()
    assert not (config.workspace / "incubus-v1-f16.gguf").exists()


def test_staged_cleanup_only_follows_a_verified_stage(tmp_path: Path) -> None:
    config = _staged_config(tmp_path)

    def failed_merge(_command: tuple[str, ...]) -> None:
        raise RuntimeError("merge failed")

    with pytest.raises(RuntimeError, match="merge failed"):
        execute_recovery_export(
            config,
            resources=lambda _path: ResourceSnapshot(100 * GIB, 20 * GIB),
            command_runner=failed_merge,
        )

    assert config.base_model.is_dir()
    assert config.adapter.is_dir()


def test_staged_quantizer_crash_keeps_only_the_f16_resume_point(tmp_path: Path) -> None:
    config = _staged_config(tmp_path)
    commands = FakeCommands(crash_quantize_once=True)

    with pytest.raises(RuntimeError, match="disconnect"):
        execute_recovery_export(
            config,
            resources=lambda _path: ResourceSnapshot(100 * GIB, 20 * GIB),
            command_runner=commands,
        )

    assert not config.base_model.exists()
    assert not (config.workspace / "merged-safetensors").exists()
    assert (config.workspace / "incubus-v1-f16.gguf").read_bytes() == b"GGUF-f16-model"
    state = json.loads((config.workspace / "recovery-state.json").read_text())
    assert set(state["stages"]) == {"merge", "convert"}

    config.base_model.mkdir()
    (config.base_model / "model.safetensors").write_bytes(b"base-weights")
    (config.base_model / "config.json").write_text("{}", encoding="utf-8")
    resumed_config = RecoveryExportConfig.create(
        run_id=config.run_id,
        base_model=config.base_model,
        adapter=config.adapter,
        workspace=config.workspace,
        merge_script=config.merge_script,
        convert_script=config.convert_script,
        quantize_binary=config.quantize_binary,
        minimum_ram_bytes=config.minimum_ram_bytes,
        minimum_free_disk_bytes=config.minimum_free_disk_bytes,
        minimum_output_bytes=config.minimum_output_bytes,
        maximum_output_bytes=config.maximum_output_bytes,
        reclaim_base_after_merge=True,
        reclaim_intermediates=True,
    )
    resumed = execute_recovery_export(
        resumed_config,
        resources=lambda _path: ResourceSnapshot(100 * GIB, 20 * GIB),
        command_runner=commands,
    )
    assert resumed.artifact.read_bytes() == b"GGUF-q5-model"
    assert len(commands.calls) == 4


def test_tampered_completed_artifact_is_requantized(tmp_path: Path) -> None:
    config = _config(tmp_path)
    commands = FakeCommands()
    result = execute_recovery_export(config, resources=_ample_resources, command_runner=commands)
    result.artifact.write_bytes(b"GGUF-tampered")

    repaired = execute_recovery_export(config, resources=_ample_resources, command_runner=commands)

    assert len(commands.calls) == 4
    assert repaired.artifact.read_bytes() == b"GGUF-q5-model"
    assert repaired.artifact_sha256 == _sha(repaired.artifact)


def test_stale_workspace_without_state_is_discarded_and_rebuilt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.workspace.mkdir(mode=0o700)
    merged = config.workspace / "merged-safetensors"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")
    (merged / "model.safetensors").write_bytes(b"stale-merged")
    (config.workspace / "incubus-v1-f16.gguf").write_bytes(b"GGUF-stale-f16")
    (config.workspace / "metaflora-incubus-v1.gguf").write_bytes(b"GGUF-stale-q5")
    commands = FakeCommands()

    result = execute_recovery_export(config, resources=_ample_resources, command_runner=commands)

    assert len(commands.calls) == 3
    assert result.artifact.read_bytes() == b"GGUF-q5-model"
    state = json.loads((config.workspace / "recovery-state.json").read_text())
    assert set(state["stages"]) == {"merge", "convert", "quantize"}


def test_output_without_matching_stage_record_is_not_adopted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initial_commands = FakeCommands()
    execute_recovery_export(config, resources=_ample_resources, command_runner=initial_commands)
    state_path = config.workspace / "recovery-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    del state["stages"]["quantize"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    commands = FakeCommands()

    result = execute_recovery_export(config, resources=_ample_resources, command_runner=commands)

    assert len(commands.calls) == 1
    assert commands.calls[0][-1] == "Q5_K_M"
    assert result.artifact.read_bytes() == b"GGUF-q5-model"


def test_mismatched_recovery_state_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.workspace.mkdir(mode=0o700)
    (config.workspace / "recovery-state.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "different-run", "stages": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ExportRecoveryError, match="run identity"):
        execute_recovery_export(
            config,
            resources=lambda _path: ResourceSnapshot(100 * GIB, 100 * GIB),
            command_runner=FakeCommands(),
        )


def test_config_rejects_invalid_identity_adapter_workspace_and_limits(tmp_path: Path) -> None:
    valid = _config(tmp_path)
    common = {
        "base_model": valid.base_model,
        "adapter": valid.adapter,
        "workspace": tmp_path / "another-workspace",
        "merge_script": valid.merge_script,
        "convert_script": valid.convert_script,
        "quantize_binary": valid.quantize_binary,
        "minimum_output_bytes": 8,
        "maximum_output_bytes": 1024,
    }
    with pytest.raises(ExportRecoveryError, match="identity"):
        RecoveryExportConfig.create(run_id="NO", **common)
    (valid.adapter / "adapter_config.json").unlink()
    with pytest.raises(ExportRecoveryError, match="adapter"):
        RecoveryExportConfig.create(run_id="incubus-v1-refine-001", **common)
    (valid.adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    workspace_file = tmp_path / "workspace-file"
    workspace_file.write_bytes(b"not-a-directory")
    with pytest.raises(ExportRecoveryError, match="workspace"):
        RecoveryExportConfig.create(
            run_id="incubus-v1-refine-001", **{**common, "workspace": workspace_file}
        )
    with pytest.raises(ExportRecoveryError, match="minimum RAM"):
        RecoveryExportConfig.create(
            run_id="incubus-v1-refine-001", **common, minimum_ram_bytes=False
        )
    with pytest.raises(ExportRecoveryError, match="size range"):
        RecoveryExportConfig.create(
            run_id="incubus-v1-refine-001",
            **{**common, "minimum_output_bytes": 2048, "maximum_output_bytes": 1024},
        )

    nested_adapter = valid.base_model / "nested-adapter"
    nested_adapter.mkdir()
    (nested_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (nested_adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (valid.base_model.parent / ".base.incubus-disposable").write_text(
        str(valid.base_model), encoding="utf-8"
    )
    with pytest.raises(ExportRecoveryError, match="protected recovery data"):
        RecoveryExportConfig.create(
            run_id="incubus-v1-refine-001",
            **{**common, "adapter": nested_adapter},
            reclaim_base_after_merge=True,
        )
    nested_base = valid.adapter / "nested-base"
    nested_base.mkdir()
    (nested_base / "config.json").write_text("{}", encoding="utf-8")
    (nested_base / "model.safetensors").write_bytes(b"base")
    (valid.adapter / ".nested-base.incubus-disposable").write_text(
        str(nested_base.resolve()), encoding="utf-8"
    )
    with pytest.raises(ExportRecoveryError, match="protected recovery data"):
        RecoveryExportConfig.create(
            run_id="incubus-v1-refine-001",
            **{**common, "base_model": nested_base},
            reclaim_base_after_merge=True,
        )
    with pytest.raises(ExportRecoveryError, match="flag"):
        RecoveryExportConfig.create(
            run_id="incubus-v1-refine-001",
            **common,
            reclaim_intermediates=1,
        )
    overlapping_workspace = valid.base_model.parent
    with pytest.raises(ExportRecoveryError, match="overlaps"):
        RecoveryExportConfig.create(
            run_id="incubus-v1-refine-001",
            **{**common, "workspace": overlapping_workspace},
        )


def test_corrupt_state_and_symlink_workspace_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.workspace.mkdir(mode=0o700)
    (config.workspace / "recovery-state.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ExportRecoveryError, match="unreadable"):
        execute_recovery_export(config, resources=_ample_resources, command_runner=FakeCommands())

    linked_workspace = tmp_path / "workspace-link"
    linked_workspace.symlink_to(config.workspace, target_is_directory=True)
    with pytest.raises(ExportRecoveryError, match="workspace"):
        RecoveryExportConfig.create(
            run_id=config.run_id,
            base_model=config.base_model,
            adapter=config.adapter,
            workspace=linked_workspace,
            merge_script=config.merge_script,
            convert_script=config.convert_script,
            quantize_binary=config.quantize_binary,
            minimum_output_bytes=8,
            maximum_output_bytes=1024,
        )


def test_reclamation_requires_a_bound_marker_and_safe_workspace(tmp_path: Path) -> None:
    valid = _config(tmp_path)
    common = {
        "run_id": valid.run_id,
        "base_model": valid.base_model,
        "adapter": valid.adapter,
        "workspace": tmp_path / "safe-workspace",
        "merge_script": valid.merge_script,
        "convert_script": valid.convert_script,
        "quantize_binary": valid.quantize_binary,
        "minimum_output_bytes": 8,
        "maximum_output_bytes": 1024,
        "reclaim_base_after_merge": True,
    }
    marker = valid.base_model.parent / ".base.incubus-disposable"
    with pytest.raises(ExportRecoveryError, match="marker is missing"):
        RecoveryExportConfig.create(**common)
    marker.write_text(str(tmp_path / "different-base"), encoding="utf-8")
    with pytest.raises(ExportRecoveryError, match="marker is invalid"):
        RecoveryExportConfig.create(**common)
    marker.write_text(str(valid.base_model), encoding="utf-8")
    with pytest.raises(ExportRecoveryError, match="workspace is unsafe"):
        RecoveryExportConfig.create(**{**common, "workspace": Path.home()})


def test_kaggle_notebook_is_recovery_only_and_resource_guarded() -> None:
    notebook = Path("notebooks/metaflora-incubus-kaggle-recover-export.ipynb")
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )

    assert len(payload["cells"]) == 2
    assert "scripts/kaggle_recover_export.py" in code
    assert "incubus-v1-refine-001" in code
    assert "Q5_K_M" in code
    assert "DPOTrainer" not in code
    assert "SFTTrainer" not in code
    assert "INCUBUS_BASE_MODEL_PATH" in code
    assert "INCUBUS_ADAPTER_PATH" in code
    assert '"--upload-repo", os.environ["INCUBUS_CHECKPOINT_LOCATION"]' in code
    assert '"--upload-branch", os.environ["INCUBUS_CHECKPOINT_BRANCH"]' in code
    assert 're.fullmatch(r"[0-9a-f]{40}", revision)' in code
    assert code.index('revision = secrets.get_secret("INCUBUS_CODE_REVISION")') < code.index(
        "shutil.rmtree(repository, ignore_errors=True)"
    )
    assert code.index("shutil.rmtree(repository, ignore_errors=True)") < code.index(
        'subprocess.run(["git", "init", str(repository)]'
    )
    assert 'git", "-C", str(repository), "rev-parse", "HEAD' in code
    assert "checked_out != revision" in code
    assert code.index('"uninstall", "-y", "torchvision", "torchaudio", "torchao"') < code.index(
        '"install", "--require-hashes"'
    )
    assert '"-e", str(repository)' not in code
    assert 'runtime_environment["PYTHONPATH"]' in code
    assert "env=runtime_environment" in code
    assert '"--reclaim-base-after-merge"' in code
    assert '"--reclaim-intermediates"' in code
    assert "reclaim_downloaded_base = False" in code
    assert "if reclaim_downloaded_base:" in code
    assert ".incubus-disposable" in code


def test_recovery_lock_installs_the_adapter_merge_runtime() -> None:
    requirements = Path("requirements/recovery.in").read_text(encoding="utf-8")

    assert "peft==0.19.1" in requirements
    assert "transformers==5.13.1" in requirements


class FakePrivateCandidateStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def ensure_private(self, *, repo_id: str, branch: str) -> None:
        self.calls.append(("ensure_private", {"repo_id": repo_id, "branch": branch}))

    def upload_candidate(self, **kwargs: object) -> str:
        self.calls.append(("upload_candidate", dict(kwargs)))
        return "a" * 40

    def verify_candidate(self, **kwargs: object) -> bool:
        self.calls.append(("verify_candidate", dict(kwargs)))
        return True

    def upload_receipt(self, **kwargs: object) -> str:
        self.calls.append(("upload_receipt", dict(kwargs)))
        return "b" * 40

    def verify_receipt(self, **kwargs: object) -> bool:
        self.calls.append(("verify_receipt", dict(kwargs)))
        return True


def test_completed_candidate_is_uploaded_with_immutable_private_evidence(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    export = execute_recovery_export(
        config, resources=_ample_resources, command_runner=FakeCommands()
    )
    store = FakePrivateCandidateStore()

    uploaded = upload_private_candidate(
        config=config,
        result=export,
        repo_id="metaflora/incubus-checkpoints",
        branch="incubus-training-v1",
        store=store,
    )

    assert isinstance(uploaded, PrivateCandidateUploadResult)
    assert uploaded.artifact_revision == "a" * 40
    assert uploaded.evidence_revision == "b" * 40
    assert uploaded.remote_prefix == (
        f"runs/{config.run_id}/exports/q5-k-m/{export.artifact_sha256}"
    )
    assert [name for name, _kwargs in store.calls] == [
        "ensure_private",
        "upload_candidate",
        "verify_candidate",
        "upload_receipt",
        "verify_receipt",
    ]
    upload = store.calls[1][1]
    assert upload["folder"] == config.workspace
    assert upload["filenames"] == (
        "candidate-export.json",
        "candidate-sha256.txt",
        "metaflora-incubus-v1.gguf",
    )
    assert str(upload["remote_prefix"]).endswith(export.artifact_sha256)
    verification = store.calls[2][1]
    assert verification["revision"] == "a" * 40
    assert verification["artifact_sha256"] == export.artifact_sha256
    assert verification["artifact_size_bytes"] == export.artifact_size_bytes
    receipt = json.loads(uploaded.receipt.read_text(encoding="utf-8"))
    assert receipt["artifact_revision"] == "a" * 40
    assert receipt["artifact_sha256"] == export.artifact_sha256
    assert receipt["release_ready"] is False
    receipt_upload = store.calls[3][1]
    assert receipt_upload["parent_revision"] == "a" * 40
    assert receipt_upload["path_in_repo"] == (
        "runs/incubus-v1-refine-001/exports/candidate-upload-receipt.json"
    )
    receipt_verification = store.calls[4][1]
    assert receipt_verification["revision"] == "b" * 40
    assert receipt_verification["expected_sha256"] == _sha(uploaded.receipt)


def test_private_upload_fails_closed_if_remote_snapshot_cannot_be_verified(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    export = execute_recovery_export(
        config, resources=_ample_resources, command_runner=FakeCommands()
    )
    store = FakePrivateCandidateStore()
    store.verify_candidate = lambda **_kwargs: False  # type: ignore[method-assign]

    with pytest.raises(ExportRecoveryError, match="verification"):
        upload_private_candidate(
            config=config,
            result=export,
            repo_id="metaflora/incubus-checkpoints",
            branch="incubus-training-v1",
            store=store,
        )
