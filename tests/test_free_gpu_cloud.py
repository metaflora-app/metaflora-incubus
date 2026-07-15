from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import sys
import types
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from metaflora_incubus.cloud_bootstrap import decrypt_cloud_bootstrap, install_cloud_bootstrap
from metaflora_incubus.cloud_training import (
    FIVE_GIB,
    CheckpointBackend,
    CloudConstraintError,
    CloudExecutionPlan,
    FreeGpuProfile,
    GoogleDriveCheckpointStore,
    HuggingFacePrivateCheckpointStore,
    RemoteCheckpointTarget,
    authorize_publication,
    cloud_disk_budget,
    load_cloud_config,
    publish_after_eval_gates,
    validate_cloud_disk_preflight,
)
from metaflora_incubus.cloud_training_runtime import (
    _benchmark_final_gguf,
    _checkout_pinned_revision,
    _hidden_huggingface_credentials,
    _latest_checkpoint,
    _prepare_ephemeral_workspace,
    _require_cached_huggingface_auth,
    _required,
    _required_revision,
    _run,
    _select_lora_targets,
    _select_model_loader_kind,
    _third_party_environment,
    _verify_checkpoint_manifest,
    _write_checkpoint_manifest,
)
from metaflora_incubus.huggingface_publication import (
    PublicationBlocker,
    PublicationDecision,
)

CONFIG_PATH = Path("configs/cloud/free-gpu-v1.json")
NOTEBOOK_PATH = Path("notebooks/metaflora-incubus-free-gpu.ipynb")
ENCRYPTED_BOOTSTRAP_PATH = Path("configs/cloud/bootstrap-v1.enc")
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "incubus_test_run_free_gpu", Path("scripts/run_free_gpu.py")
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
free_gpu_runner = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(free_gpu_runner)


@pytest.mark.parametrize("failure_stage", ("revision", "vram", "plan"))
def test_runner_reports_every_early_execution_failure_after_argument_parsing(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    class EarlyFailure(RuntimeError):
        pass

    captured: dict[str, object] = {}

    def reporting_boundary(operation, **kwargs):
        captured.update(kwargs)
        return operation()

    monkeypatch.setattr(free_gpu_runner, "run_with_failure_reporting", reporting_boundary)
    monkeypatch.setattr(free_gpu_runner, "detect_code_revision", lambda: "a" * 40)
    monkeypatch.setattr(free_gpu_runner, "detect_vram_bytes", lambda: 16 * 1024**3)
    if failure_stage == "revision":
        monkeypatch.setattr(
            free_gpu_runner,
            "detect_code_revision",
            lambda: (_ for _ in ()).throw(EarlyFailure("revision failed")),
        )
    elif failure_stage == "vram":
        monkeypatch.setattr(
            free_gpu_runner,
            "detect_vram_bytes",
            lambda: (_ for _ in ()).throw(EarlyFailure("vram failed")),
        )
    else:
        monkeypatch.setattr(
            free_gpu_runner.CloudExecutionPlan,
            "create",
            lambda **kwargs: (_ for _ in ()).throw(EarlyFailure("plan failed")),
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_free_gpu.py",
            "--execute",
            "--run-id",
            "incubus-v1-run",
            "--parameter-count",
            "4659865088",
            "--checkpoint-backend",
            "hf_private_branch",
            "--checkpoint-location",
            "private-owner/private-checkpoints",
            "--checkpoint-branch",
            "incubus-training-v1",
        ],
    )

    with pytest.raises(EarlyFailure, match=failure_stage):
        free_gpu_runner.main()

    assert captured["run_id"] == "incubus-v1-run"
    assert captured["phase"] == "post-bootstrap-execution"
    assert captured["code_revision"] == ("unknown" if failure_stage == "revision" else "a" * 40)


def test_free_gpu_profile_is_honest_about_t4_and_rejects_nine_billion_parameters() -> None:
    profile = FreeGpuProfile.default()

    assert profile.max_parameter_count == 7_500_000_000
    assert profile.max_sequence_length == 2048
    assert profile.load_in_4bit is True
    assert profile.quantization_type == "nf4"
    assert profile.final_gguf_max_bytes == FIVE_GIB
    profile.validate(parameter_count=7_000_000_000, vram_bytes=15 * 1024**3)

    with pytest.raises(CloudConstraintError, match="parameter"):
        profile.validate(parameter_count=9_000_000_000, vram_bytes=16 * 1024**3)
    with pytest.raises(CloudConstraintError, match="VRAM"):
        profile.validate(parameter_count=7_000_000_000, vram_bytes=12 * 1024**3)
    with pytest.raises(FrozenInstanceError):
        profile.max_sequence_length = 4096  # type: ignore[misc]


def test_cloud_config_is_brand_only_secret_free_and_uses_ephemeral_workspace() -> None:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    config = load_cloud_config(CONFIG_PATH)

    assert config.product_id == "metaflora-incubus-v1"
    assert config.workspace == Path("/content/incubus-work")
    assert config.public_repo_id == "metaflora/incubus"
    assert config.profile == FreeGpuProfile.default()
    assert "build_input_repo_id" not in raw.casefold()
    assert "hf_" not in raw.casefold()
    assert "token" not in json.loads(raw)


def test_checkpoint_target_is_remote_private_and_never_main_branch() -> None:
    hf = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location="private-owner/private-checkpoints",
        branch="incubus-training-v1",
    )
    drive = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.GOOGLE_DRIVE,
        location="/content/drive/MyDrive/metaflora-incubus/checkpoints",
        branch=None,
    )

    assert hf.branch == "incubus-training-v1"
    assert drive.location.startswith("/content/drive/MyDrive/")
    with pytest.raises(CloudConstraintError, match="main"):
        RemoteCheckpointTarget.create(
            backend=CheckpointBackend.HF_PRIVATE_BRANCH,
            location="private-owner/private-checkpoints",
            branch="main",
        )
    with pytest.raises(CloudConstraintError, match="private branch"):
        RemoteCheckpointTarget.create(
            backend=CheckpointBackend.HF_PRIVATE_BRANCH,
            location="metaflora/incubus",
            branch="training",
        )
    with pytest.raises(CloudConstraintError, match="MyDrive"):
        RemoteCheckpointTarget.create(
            backend=CheckpointBackend.GOOGLE_DRIVE,
            location="/content/checkpoints",
            branch=None,
        )


def test_plan_keeps_intermediate_weights_off_the_users_mac() -> None:
    config = load_cloud_config(CONFIG_PATH)
    target = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location="private-owner/private-checkpoints",
        branch="incubus-training-v1",
    )
    plan = CloudExecutionPlan.create(
        config=config,
        checkpoint_target=target,
        run_id="run-20260714-001",
        parameter_count=7_000_000_000,
        vram_bytes=16 * 1024**3,
    )

    assert plan.workspace == Path("/content/incubus-work/run-20260714-001")
    assert plan.local_retention is False
    assert plan.resume_enabled is True
    assert plan.training_mode == "qlora_nf4"
    assert plan.post_training_steps == (
        "convert_base_to_f16_gguf",
        "convert_adapter_to_gguf",
        "merge_gguf_in_cloud",
        "quantize_q5_k_m",
        "run_candidate_benchmark",
        "sync_private_evidence",
        "delete_ephemeral_workspace",
    )


def test_final_cloud_artifact_runs_pinned_local_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(load_cloud_config(CONFIG_PATH), workspace=tmp_path / "cloud")
    target = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location="private-owner/private-checkpoints",
        branch="incubus-training-v1",
    )
    plan = CloudExecutionPlan.create(
        config=config,
        checkpoint_target=target,
        run_id="benchmark-final",
        parameter_count=7_000_000_000,
        vram_bytes=16 * 1024**3,
    )
    server = plan.workspace / "llama.cpp" / "build" / "bin" / "llama-server"
    server.parent.mkdir(parents=True)
    server.write_bytes(b"server")
    artifact = tmp_path / "final.gguf"
    artifact.write_bytes(b"GGUFmodel")
    captured: dict[str, object] = {}

    def fake_run(runner_config):
        captured["config"] = runner_config
        return {"artifact_sha256": runner_config.model_sha256, "case_count": 48}

    monkeypatch.setattr(
        "metaflora_incubus.gguf_benchmark_runner.run_gguf_benchmark",
        fake_run,
    )

    evidence = _benchmark_final_gguf(plan=plan, artifact=artifact)

    runner_config = captured["config"]
    assert runner_config.server_binary == server
    assert runner_config.model_path == artifact
    assert runner_config.cases_path.name == "gguf-v1-cases.jsonl"
    assert runner_config.seed == 4242
    assert evidence["case_count"] == 48


def test_cloud_disk_preflight_calculates_full_peak_and_fails_fast() -> None:
    config = load_cloud_config(CONFIG_PATH)
    target = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location="private-owner/private-checkpoints",
        branch="incubus-training-v1",
    )
    plan = CloudExecutionPlan.create(
        config=config,
        checkpoint_target=target,
        run_id="run-disk-preflight",
        parameter_count=7_000_000_000,
        vram_bytes=16 * 1024**3,
    )

    budget = cloud_disk_budget(plan)

    assert budget.source_download_bytes == 14_000_000_000
    assert budget.build_intermediate_bytes == 28_000_000_000
    assert budget.final_artifact_bytes == FIVE_GIB
    assert budget.reserve_bytes == 8 * 1024**3
    assert budget.required_bytes == (
        budget.source_download_bytes
        + budget.build_intermediate_bytes
        + budget.final_artifact_bytes
        + budget.reserve_bytes
    )
    validate_cloud_disk_preflight(plan, available_bytes=budget.required_bytes)
    with pytest.raises(CloudConstraintError, match="disk preflight"):
        validate_cloud_disk_preflight(plan, available_bytes=budget.required_bytes - 1)


def test_repeated_run_cleans_only_ephemeral_workspace_then_restores_signed_checkpoint(
    tmp_path: Path,
) -> None:
    config = replace(load_cloud_config(CONFIG_PATH), workspace=tmp_path / "ephemeral-root")
    target = RemoteCheckpointTarget(
        backend=CheckpointBackend.GOOGLE_DRIVE,
        location=str(tmp_path / "remote"),
        branch=None,
    )
    plan = CloudExecutionPlan.create(
        config=config,
        checkpoint_target=target,
        run_id="run-safe-repeat",
        parameter_count=7_000_000_000,
        vram_bytes=16 * 1024**3,
    )
    plan.workspace.mkdir(parents=True)
    (plan.workspace / "stale-download.bin").write_bytes(b"stale")
    sibling = config.workspace / "different-run"
    sibling.mkdir()
    (sibling / "must-survive").write_text("keep", encoding="utf-8")

    binding = {"run_id": plan.run_id, "code_revision": "a" * 40}
    key = "checkpoint-authentication-key-32bytes"
    remote_checkpoint = Path(target.location) / plan.run_id
    (remote_checkpoint / "sft" / "checkpoint-10").mkdir(parents=True)
    (remote_checkpoint / "sft" / "checkpoint-10" / "state.json").write_text("{}", encoding="utf-8")
    _write_checkpoint_manifest(remote_checkpoint, binding=binding, key=key)
    store = GoogleDriveCheckpointStore(target, plan.run_id)

    restored = _prepare_ephemeral_workspace(
        plan=plan,
        store=store,
        binding=binding,
        checkpoint_key=key,
    )

    assert restored == plan.workspace / "checkpoints"
    assert not (plan.workspace / "stale-download.bin").exists()
    assert (restored / "sft" / "checkpoint-10" / "state.json").is_file()
    assert (sibling / "must-survive").read_text(encoding="utf-8") == "keep"
    assert (remote_checkpoint / "sft" / "checkpoint-10" / "state.json").is_file()


def test_disk_preflight_runs_after_safe_cleanup_but_before_remote_restore(
    tmp_path: Path,
) -> None:
    config = replace(load_cloud_config(CONFIG_PATH), workspace=tmp_path / "ephemeral-root")
    target = RemoteCheckpointTarget(
        backend=CheckpointBackend.GOOGLE_DRIVE,
        location=str(tmp_path / "remote"),
        branch=None,
    )
    plan = CloudExecutionPlan.create(
        config=config,
        checkpoint_target=target,
        run_id="run-preflight-order",
        parameter_count=7_000_000_000,
        vram_bytes=16 * 1024**3,
    )
    plan.workspace.mkdir(parents=True)
    stale = plan.workspace / "stale"
    stale.write_bytes(b"delete only after scope validation")

    class TrackingStore:
        restore_calls = 0

        def restore(self, destination: Path) -> Path | None:
            del destination
            self.restore_calls += 1
            raise AssertionError("restore must not run after failed disk preflight")

    store = TrackingStore()
    with pytest.raises(CloudConstraintError, match="disk preflight"):
        _prepare_ephemeral_workspace(
            plan=plan,
            store=store,
            binding={"run_id": plan.run_id},
            checkpoint_key="checkpoint-authentication-key-32bytes",
            before_restore=lambda: validate_cloud_disk_preflight(plan, available_bytes=0),
        )

    assert store.restore_calls == 0
    assert not stale.exists()
    assert plan.workspace.is_dir()


@pytest.mark.parametrize("size_bytes", (3 * 1024**3, FIVE_GIB))
def test_publication_authorization_requires_passing_gates_and_at_most_five_gib(
    tmp_path: Path, size_bytes: int
) -> None:
    artifact = tmp_path / "metaflora-incubus-v1.gguf"
    artifact.touch()
    with artifact.open("r+b") as handle:
        handle.truncate(size_bytes)

    authorization = authorize_publication(
        artifact_path=artifact,
        gate_decision=PublicationDecision(True, ()),
        repo_id="metaflora/incubus",
    )

    assert authorization.repo_id == "metaflora/incubus"
    assert authorization.artifact_size_bytes == size_bytes
    assert authorization.allowed is True


def test_publication_is_fail_closed_before_any_direct_upload(tmp_path: Path) -> None:
    artifact = tmp_path / "metaflora-incubus-v1.gguf"
    artifact.touch()
    with artifact.open("r+b") as handle:
        handle.truncate(FIVE_GIB + 1)

    with pytest.raises(CloudConstraintError, match="5 GiB"):
        authorize_publication(
            artifact_path=artifact,
            gate_decision=PublicationDecision(True, ()),
            repo_id="metaflora/incubus",
        )

    artifact.write_bytes(b"too small")
    with pytest.raises(CloudConstraintError, match="3 GiB"):
        authorize_publication(
            artifact_path=artifact,
            gate_decision=PublicationDecision(True, ()),
            repo_id="metaflora/incubus",
        )


def test_direct_hub_upload_is_not_called_when_eval_gates_fail(tmp_path: Path) -> None:
    artifact = tmp_path / "metaflora-incubus-v1.gguf"
    artifact.write_bytes(b"candidate")
    failed = PublicationDecision(False, (PublicationBlocker("target_miss", "coding"),))

    class NeverUploader:
        calls = 0

        def __getattr__(self, name: str):
            del name
            self.calls += 1
            raise AssertionError("uploader must not be touched")

    uploader = NeverUploader()
    with pytest.raises(CloudConstraintError, match="eval gates"):
        publish_after_eval_gates(
            bundle=tmp_path,
            artifact_path=artifact,
            gate_decision=failed,
            signature_verifier=lambda purpose, payload, signature: True,
            uploader=uploader,
            prohibited_identifiers=("private-build-input",),
        )
    assert uploader.calls == 0

    artifact.write_bytes(b"small")
    failed = PublicationDecision(False, (PublicationBlocker("target_miss", "coding"),))
    with pytest.raises(CloudConstraintError, match="eval gates"):
        authorize_publication(
            artifact_path=artifact,
            gate_decision=failed,
            repo_id="metaflora/incubus",
        )


def test_one_click_notebook_uses_cloud_secrets_and_no_local_mac_paths() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join(line for cell in notebook["cells"] for line in cell.get("source", []))

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert "google.colab" in source
    assert 'userdata.get("INCUBUS_BOOTSTRAP")' in source
    assert "notebook access is disabled" in source
    assert 'RuntimeError("INCUBUS_BOOTSTRAP is empty")' in source
    assert "decrypt_cloud_bootstrap(encrypted_bootstrap, encoded_bootstrap)" in source
    assert "base64.b64decode" not in source
    assert ENCRYPTED_BOOTSTRAP_PATH.is_file()
    assert "scripts/run_free_gpu.py" in source
    assert "--execute" in source
    assert "--require-hashes" in source
    assert source.index('os.chdir("/content")') < source.index(
        "shutil.rmtree(repository, ignore_errors=True)"
    )
    assert "shutil.rmtree(repository, ignore_errors=True)" in source
    assert "sys.path.insert(0, source_root)" in source
    assert "requirements/cloud-linux.lock" in source
    revisions = re.findall(r'trusted_code_revision = "([0-9a-f]{40})"', source)
    assert revisions == ["ec160e2d3f549350d6afd6cc2ab25791ae227661"]
    assert 'git", "clone' not in source
    assert 'rev-parse", "HEAD' in source
    assert "HF_TOKEN" not in source
    assert 'os.environ["INCUBUS_PARAMETER_COUNT"]' in source
    assert "/Users/" not in source
    assert "build_input_repo_id=" not in source.casefold()


def test_single_bootstrap_restores_cached_auth_and_seven_generic_values(tmp_path: Path) -> None:
    values = {
        "INCUBUS_CHECKPOINT_HMAC_KEY": "checkpoint-authentication-key-32bytes",
        "INCUBUS_SOURCE_REPO": "private/source",
        "INCUBUS_SOURCE_REVISION": "a" * 40,
        "INCUBUS_DATASET_REPO": "private/dataset",
        "INCUBUS_DATASET_REVISION": "b" * 40,
        "INCUBUS_DATASET_SHA256": "c" * 64,
        "INCUBUS_PARAMETER_COUNT": "7000000000",
    }
    payload = json.dumps(
        {
            "hf_token": "cached-access-token",
            "hf_stored_tokens": "[account]\nhf_token = cached-access-token\n",
            "environment": values,
        }
    ).encode()
    environment = {"HF_TOKEN": "must-be-removed"}

    parameter_count = install_cloud_bootstrap(payload, home=tmp_path, environment=environment)

    assert parameter_count == 7_000_000_000
    assert environment == values
    cache = tmp_path / ".cache" / "huggingface"
    assert (cache / "token").read_text() == "cached-access-token"
    assert "cached-access-token" in (cache / "stored_tokens").read_text()
    assert (cache / "token").stat().st_mode & 0o777 == 0o600
    assert (cache / "stored_tokens").stat().st_mode & 0o777 == 0o600


def test_short_key_decrypts_opaque_bootstrap_and_tampering_fails() -> None:
    key = os.urandom(32)
    nonce = os.urandom(12)
    payload = b'{"safe":"synthetic"}'
    encrypted = (
        b"INCUBUS1"
        + nonce
        + AESGCM(key).encrypt(nonce, payload, b"metaflora-incubus-cloud-bootstrap-v1")
    )
    encoded_key = base64.urlsafe_b64encode(key).decode()

    assert decrypt_cloud_bootstrap(encrypted, encoded_key) == payload
    with pytest.raises(CloudConstraintError, match="did not decrypt"):
        decrypt_cloud_bootstrap(encrypted[:-1] + bytes([encrypted[-1] ^ 1]), encoded_key)


@pytest.mark.parametrize("parameter_count", ("0", "7500000001", "not-an-int"))
def test_bootstrap_rejects_invalid_parameter_count(tmp_path: Path, parameter_count: str) -> None:
    values = {
        "INCUBUS_CHECKPOINT_HMAC_KEY": "checkpoint-authentication-key-32bytes",
        "INCUBUS_SOURCE_REPO": "private/source",
        "INCUBUS_SOURCE_REVISION": "a" * 40,
        "INCUBUS_DATASET_REPO": "private/dataset",
        "INCUBUS_DATASET_REVISION": "b" * 40,
        "INCUBUS_DATASET_SHA256": "c" * 64,
        "INCUBUS_PARAMETER_COUNT": parameter_count,
    }
    payload = json.dumps(
        {"hf_token": "token", "hf_stored_tokens": "stored", "environment": values}
    ).encode()
    with pytest.raises(CloudConstraintError, match="parameter count"):
        install_cloud_bootstrap(payload, home=tmp_path, environment={})


def test_cloud_huggingface_clients_use_cached_auth_and_early_whoami(
    monkeypatch,
) -> None:
    constructor_tokens: list[object] = []
    whoami_calls = 0

    class FakeHfApi:
        def __init__(self, *, token=None):
            constructor_tokens.append(token)

        def whoami(self):
            nonlocal whoami_calls
            whoami_calls += 1
            return {"name": "cached-user"}

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))

    _require_cached_huggingface_auth()
    target = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location="private-owner/private-checkpoints",
        branch="incubus-training-v1",
    )
    HuggingFacePrivateCheckpointStore(target, "run-cached-auth")

    assert whoami_calls == 1
    assert constructor_tokens == [None, None]


def test_native_process_environment_and_benchmark_hide_cached_hub_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    token = cache / "token"
    stored = cache / "stored_tokens"
    token.write_text("access-token", encoding="utf-8")
    stored.write_text("refresh-token", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "must-not-pass")
    monkeypatch.setenv("HF_HOME", str(cache))

    environment = _third_party_environment()
    assert "HOME" not in environment
    assert "HF_TOKEN" not in environment
    assert "HF_HOME" not in environment

    with _hidden_huggingface_credentials(tmp_path):
        assert not token.exists()
        assert not stored.exists()

    assert token.read_text(encoding="utf-8") == "access-token"
    assert stored.read_text(encoding="utf-8") == "refresh-token"
    assert token.stat().st_mode & 0o777 == 0o600
    assert stored.stat().st_mode & 0o777 == 0o600


def test_drive_checkpoint_copy_and_runtime_resume_selection(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    target = RemoteCheckpointTarget(
        backend=CheckpointBackend.GOOGLE_DRIVE,
        location=str(remote),
        branch=None,
    )
    store = GoogleDriveCheckpointStore(target, "run-001")
    local = tmp_path / "local"
    (local / "checkpoint-2").mkdir(parents=True)
    (local / "checkpoint-2" / "state.json").write_text("{}")
    (local / "checkpoint-10").mkdir()
    (local / "checkpoint-10" / "state.json").write_text("{}")

    store.sync(local)
    restored = store.restore(tmp_path / "restored")

    assert restored == tmp_path / "restored"
    assert _latest_checkpoint(restored).name == "checkpoint-10"
    assert _required({"SECRET": "value"}, "SECRET") == "value"
    with pytest.raises(CloudConstraintError, match="secret name"):
        _required({}, "SECRET")
    assert _required_revision({"REV": "a" * 40}, "REV") == "a" * 40
    with pytest.raises(CloudConstraintError, match="40-hex"):
        _required_revision({"REV": "main"}, "REV")


def test_cloud_runtime_selects_text_or_multimodal_loader_from_pinned_config() -> None:
    assert _select_model_loader_kind(("DenseForCausalLM",)) == "causal_lm"
    assert _select_model_loader_kind(("DenseForConditionalGeneration",)) == "image_text_to_text"
    with pytest.raises(CloudConstraintError, match="architecture"):
        _select_model_loader_kind(())


def test_cloud_runtime_targets_language_layers_without_touching_vision_modules() -> None:
    targets = _select_lora_targets(
        (
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.mlp.down_proj",
            "model.visual.blocks.0.attn.q_proj",
            "lm_head",
        )
    )

    assert targets == (
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.self_attn.q_proj",
    )
    with pytest.raises(CloudConstraintError, match="LoRA"):
        _select_lora_targets(("model.visual.blocks.0.attn.q_proj",))


def test_checkpoint_manifest_binds_inputs_and_detects_remote_tampering(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    weights = checkpoint / "adapter_model.safetensors"
    weights.write_bytes(b"trusted adapter")
    binding = {
        "code_revision": "a" * 40,
        "config_sha256": "b" * 64,
        "dataset_revision": "c" * 40,
        "dataset_sha256": "d" * 64,
        "run_id": "incubus-v1-run",
        "source_revision": "e" * 40,
    }
    key = "checkpoint-authentication-key-32bytes"

    _write_checkpoint_manifest(checkpoint, binding=binding, key=key)
    _verify_checkpoint_manifest(checkpoint, binding=binding, key=key)

    weights.write_bytes(b"tampered adapter")
    with pytest.raises(CloudConstraintError, match="integrity"):
        _verify_checkpoint_manifest(checkpoint, binding=binding, key=key)
    with pytest.raises(CloudConstraintError, match="binding"):
        _verify_checkpoint_manifest(
            checkpoint,
            binding={**binding, "dataset_revision": "f" * 40},
            key=key,
        )


def test_cloud_config_and_authorization_reject_malformed_inputs(tmp_path: Path) -> None:
    malformed = tmp_path / "config.json"
    malformed.write_text('{"schema_version": 2}')
    with pytest.raises(CloudConstraintError, match="schema"):
        load_cloud_config(malformed)

    missing = tmp_path / "missing.gguf"
    with pytest.raises(CloudConstraintError, match="missing"):
        authorize_publication(
            artifact_path=missing,
            gate_decision=PublicationDecision(True, ()),
            repo_id="metaflora/incubus",
        )


def test_cloud_config_rejects_mutable_llama_cpp_revision(tmp_path: Path) -> None:
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    document["llama_cpp_revision"] = "main"
    mutable = tmp_path / "cloud.json"
    mutable.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CloudConstraintError, match="40-hex"):
        load_cloud_config(mutable)


def test_third_party_commands_receive_no_cloud_secrets(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return type("Result", (), {"stdout": "a" * 40 + "\n"})()

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf_top_secret")
    monkeypatch.setenv("INCUBUS_CHECKPOINT_HMAC_KEY", "checkpoint-secret")
    monkeypatch.setenv("UNRELATED_PASSWORD", "password-secret")
    monkeypatch.setattr("metaflora_incubus.cloud_training_runtime.subprocess.run", fake_run)

    _run(["cmake", "--version"])
    _checkout_pinned_revision(tmp_path, "a" * 40)

    assert calls
    for call in calls:
        environment = call["env"]
        assert environment["PATH"] == "/usr/bin"
        assert "HOME" not in environment
        assert "HF_TOKEN" not in environment
        assert "INCUBUS_CHECKPOINT_HMAC_KEY" not in environment
        assert "UNRELATED_PASSWORD" not in environment


def test_llama_cpp_checkout_must_resolve_to_the_exact_commit(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, **kwargs):
        stdout = "b" * 40 + "\n" if command[:3] == ["git", "rev-parse", "HEAD"] else ""
        return type("Result", (), {"stdout": stdout})()

    monkeypatch.setattr("metaflora_incubus.cloud_training_runtime.subprocess.run", fake_run)

    with pytest.raises(CloudConstraintError, match="checkout"):
        _checkout_pinned_revision(tmp_path, "a" * 40)
