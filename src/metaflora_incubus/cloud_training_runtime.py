"""CUDA-only QLoRA runtime used by the free-tier notebook.

Imports are intentionally lazy so planning and dry runs work on a Mac without
CUDA, bitsandbytes, or the Hub client.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from metaflora_incubus.cloud_training import (
    CheckpointBackend,
    CloudConstraintError,
    CloudExecutionPlan,
    GoogleDriveCheckpointStore,
    HuggingFacePrivateCheckpointStore,
    safe_ephemeral_cleanup,
    validate_cloud_disk_preflight,
)
from metaflora_incubus.huggingface_publication import MIN_MODEL_BYTES


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not value:
        raise CloudConstraintError(f"missing cloud secret name: {name}")
    return value


def _required_revision(environment: Mapping[str, str], name: str) -> str:
    revision = _required(environment, name)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise CloudConstraintError(f"{name} must be a pinned 40-hex revision")
    return revision


def _select_model_loader_kind(architectures: tuple[str, ...]) -> str:
    if any(name.endswith("ForConditionalGeneration") for name in architectures):
        return "image_text_to_text"
    if any(name.endswith("ForCausalLM") for name in architectures):
        return "causal_lm"
    raise CloudConstraintError("unsupported pinned model architecture")


def _select_lora_targets(module_names: tuple[str, ...]) -> tuple[str, ...]:
    suffixes = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    targets = tuple(
        sorted(
            name
            for name in module_names
            if name.endswith(suffixes)
            and "visual" not in name.casefold()
            and "vision" not in name.casefold()
        )
    )
    if not targets:
        raise CloudConstraintError("no compatible language LoRA targets were found")
    return targets


def _cast_trainable_parameters_to_fp32(model, *, torch_module) -> None:
    trainable_count = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        trainable_count += 1
        if parameter.dtype == torch_module.bfloat16:
            parameter.data = parameter.data.to(torch_module.float32)
    if trainable_count == 0:
        raise CloudConstraintError("training model has no trainable parameters")


_CHECKPOINT_MANIFEST = "incubus-checkpoint-manifest.json"
_TRAINING_CHECKPOINT_FILE = re.compile(r"(?:sft|preference)/checkpoint-[0-9]+/.+")
_TRAINING_STAGE_FILE = re.compile(r"(?:sft|preference)/.+")


def _is_prunable_training_file(name: str) -> bool:
    return _TRAINING_CHECKPOINT_FILE.fullmatch(name) is not None


def _is_missing_pruned_training_file(name: str) -> bool:
    return _TRAINING_STAGE_FILE.fullmatch(name) is not None


def _checkpoint_file_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise CloudConstraintError("checkpoint directory is invalid")
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CloudConstraintError("checkpoint integrity rejects symbolic links")
        if not path.is_file() or path.name == _CHECKPOINT_MANIFEST:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
    if not hashes:
        raise CloudConstraintError("checkpoint integrity requires files")
    return hashes


def _checkpoint_payload(*, binding: Mapping[str, str], files: Mapping[str, str]) -> bytes:
    document = {"binding": dict(sorted(binding.items())), "files": dict(sorted(files.items()))}
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _checkpoint_key(key: str) -> bytes:
    encoded = key.encode()
    if len(encoded) < 32:
        raise CloudConstraintError("checkpoint authentication key is too short")
    return encoded


def _write_checkpoint_manifest(root: Path, *, binding: Mapping[str, str], key: str) -> None:
    files = _checkpoint_file_hashes(root)
    payload = _checkpoint_payload(binding=binding, files=files)
    document = {
        "binding": dict(sorted(binding.items())),
        "files": files,
        "hmac_sha256": hmac.new(_checkpoint_key(key), payload, hashlib.sha256).hexdigest(),
        "schema_version": 1,
    }
    temporary = root / f".{_CHECKPOINT_MANIFEST}.tmp"
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(root / _CHECKPOINT_MANIFEST)


def _verify_checkpoint_manifest(
    root: Path,
    *,
    binding: Mapping[str, str],
    key: str,
    allow_prunable_training_files: bool = False,
) -> None:
    try:
        document = json.loads((root / _CHECKPOINT_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloudConstraintError("checkpoint integrity manifest is invalid") from exc
    if document.get("schema_version") != 1 or document.get("binding") != dict(binding):
        raise CloudConstraintError("checkpoint binding does not match this training run")
    recorded_files = document.get("files")
    actual_files = _checkpoint_file_hashes(root)
    if not isinstance(recorded_files, dict):
        raise CloudConstraintError("checkpoint integrity verification failed")
    missing = set(recorded_files).difference(actual_files)
    if missing and (
        not allow_prunable_training_files
        or any(not _is_missing_pruned_training_file(name) for name in missing)
    ):
        raise CloudConstraintError("checkpoint integrity verification failed")
    if any(
        actual_files[name] != recorded_files[name]
        for name in set(recorded_files).intersection(actual_files)
    ):
        raise CloudConstraintError("checkpoint integrity verification failed")
    untracked = set(actual_files).difference(recorded_files)
    if untracked and (
        not allow_prunable_training_files
        or any(not _is_prunable_training_file(name) for name in untracked)
    ):
        raise CloudConstraintError("checkpoint integrity verification failed")
    payload = _checkpoint_payload(binding=binding, files=recorded_files)
    expected = hmac.new(_checkpoint_key(key), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(document.get("hmac_sha256", "")), expected):
        raise CloudConstraintError("checkpoint integrity authentication failed")


def _authenticated_recovery_binding(
    root: Path,
    *,
    environment: Mapping[str, str],
    checkpoint_key: str,
    run_id: str,
) -> dict[str, str]:
    """Authenticate a completed checkpoint while allowing recovery-code upgrades."""

    try:
        document = json.loads((root / _CHECKPOINT_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloudConstraintError("checkpoint integrity manifest is invalid") from exc
    binding = document.get("binding")
    if not isinstance(binding, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in binding.items()
    ):
        raise CloudConstraintError("checkpoint recovery binding is invalid")
    authenticated = dict(binding)
    _verify_checkpoint_manifest(
        root,
        binding=authenticated,
        key=checkpoint_key,
        allow_prunable_training_files=True,
    )
    if authenticated.get("run_id") != run_id:
        raise CloudConstraintError("checkpoint recovery run identity does not match")
    source_repo = _required(environment, "INCUBUS_SOURCE_REPO")
    if authenticated.get("source_repo_sha256") != hashlib.sha256(source_repo.encode()).hexdigest():
        raise CloudConstraintError("checkpoint recovery source identity does not match")
    if authenticated.get("source_revision") != _required_revision(
        environment, "INCUBUS_SOURCE_REVISION"
    ):
        raise CloudConstraintError("checkpoint recovery source revision does not match")
    dataset_repo = _required(environment, "INCUBUS_DATASET_REPO")
    if (
        authenticated.get("dataset_repo_sha256")
        != hashlib.sha256(dataset_repo.encode()).hexdigest()
    ):
        raise CloudConstraintError("checkpoint recovery dataset identity does not match")
    expected_dataset_values = {
        "dataset_revision": _required_revision(environment, "INCUBUS_DATASET_REVISION"),
        "dataset_sha256": _required(environment, "INCUBUS_DATASET_SHA256"),
    }
    if any(authenticated.get(name) != value for name, value in expected_dataset_values.items()):
        raise CloudConstraintError("checkpoint recovery dataset binding does not match")
    return authenticated


_THIRD_PARTY_ENVIRONMENT_KEYS = (
    "CC",
    "CMAKE_PREFIX_PATH",
    "CPATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CXX",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "PATH",
    "PKG_CONFIG_PATH",
    "TEMP",
    "TMP",
    "TMPDIR",
)


def _third_party_environment() -> dict[str, str]:
    return {name: os.environ[name] for name in _THIRD_PARTY_ENVIRONMENT_KEYS if name in os.environ}


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True, env=_third_party_environment())


@contextmanager
def _hidden_huggingface_credentials(home: Path):
    """Remove cached Hub credentials while untrusted native code is executing."""
    from metaflora_incubus.cloud_bootstrap import _private_atomic_write

    paths = (
        home / ".cache" / "huggingface" / "token",
        home / ".cache" / "huggingface" / "stored_tokens",
    )
    try:
        contents = {path: path.read_text(encoding="utf-8") for path in paths}
    except OSError as exc:
        raise CloudConstraintError("cached Hugging Face credentials are missing") from exc
    for path in paths:
        path.unlink()
    try:
        yield
    finally:
        for path, value in contents.items():
            _private_atomic_write(path, value)


def _checkout_pinned_revision(repository: Path, revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise CloudConstraintError("llama.cpp revision must be a pinned 40-hex commit")
    _run(["git", "checkout", "--detach", revision], cwd=repository)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        env=_third_party_environment(),
    )
    if result.stdout.strip().lower() != revision:
        raise CloudConstraintError("llama.cpp checkout did not resolve to the pinned commit")


def _require_cached_huggingface_auth() -> None:
    from huggingface_hub import HfApi

    try:
        identity = HfApi(token=None).whoami()
    except Exception as exc:
        raise CloudConstraintError("cached Hugging Face authentication failed") from exc
    if not isinstance(identity, dict) or not identity.get("name"):
        raise CloudConstraintError("cached Hugging Face identity is invalid")


def _checkpoint_store(plan: CloudExecutionPlan):
    if plan.checkpoint_target.backend is CheckpointBackend.GOOGLE_DRIVE:
        return GoogleDriveCheckpointStore(plan.checkpoint_target, plan.run_id)
    return HuggingFacePrivateCheckpointStore(plan.checkpoint_target, plan.run_id)


def _prepare_ephemeral_workspace(
    *,
    plan: CloudExecutionPlan,
    store,
    binding: Mapping[str, str],
    checkpoint_key: str,
    before_restore: Callable[[], None] | None = None,
) -> Path | None:
    """Reset only this run's ephemeral directory, then restore authenticated state."""

    if plan.config.workspace.is_symlink() or plan.workspace.is_symlink():
        raise CloudConstraintError("refusing symlinked ephemeral workspace")
    if plan.workspace != plan.config.workspace / plan.run_id:
        raise CloudConstraintError("run workspace does not match its run_id")
    plan.config.workspace.mkdir(parents=True, exist_ok=True)
    if plan.workspace.exists() or plan.workspace.is_symlink():
        safe_ephemeral_cleanup(plan)
    plan.workspace.mkdir(parents=False, exist_ok=False)
    if before_restore is not None:
        before_restore()
    checkpoint_root = plan.workspace / "checkpoints"
    restore = getattr(store, "restore_recovery", store.restore)
    restored = restore(checkpoint_root)
    if restored is not None:
        _verify_checkpoint_manifest(restored, binding=binding, key=checkpoint_key)
    return restored


def _available_workspace_bytes(plan: CloudExecutionPlan) -> int:
    return shutil.disk_usage(plan.workspace).free


def _latest_checkpoint(root: Path) -> Path | None:
    candidates = tuple(root.rglob("checkpoint-*")) if root.is_dir() else ()
    valid = tuple(
        item
        for item in candidates
        if item.is_dir() and item.name.removeprefix("checkpoint-").isdigit()
    )
    return (
        max(valid, key=lambda item: int(item.name.removeprefix("checkpoint-"))) if valid else None
    )


def _mixed_dataset(load_dataset, interleave_datasets, path: Path, seed: int):
    raw = load_dataset("json", data_files=str(path), split="train")
    names = ("code", "agentic_tools", "russian_text", "english_text")
    probabilities = (0.35, 0.25, 0.20, 0.20)
    partitions = tuple(
        raw.filter(lambda row, expected=name: row["capability"] == expected) for name in names
    )
    if any(len(partition) == 0 for partition in partitions):
        raise CloudConstraintError("prepared dataset is missing a required capability")
    return interleave_datasets(
        partitions,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy="all_exhausted",
    )


@dataclass(frozen=True)
class TextPreflightReport:
    checked_records: int
    maximum_tokens: int
    max_sequence_length: int


def _message_list(row: Mapping[str, object], key: str) -> tuple[dict[str, str], ...]:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        raise CloudConstraintError(f"training text schema is invalid: {key}")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise CloudConstraintError(f"training text schema is invalid: {key}")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role.strip():
            raise CloudConstraintError(f"training text schema is invalid: {key}")
        if not isinstance(content, str) or not content.strip():
            raise CloudConstraintError(f"training text schema is invalid: {key}")
        messages.append({"role": role.strip(), "content": content})
    return tuple(messages)


def _token_count(tokenizer: object, messages: tuple[dict[str, str], ...]) -> int:
    try:
        tokens = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
        )
        count = len(tokens)
    except Exception as exc:
        raise CloudConstraintError("chat template could not tokenize training text") from exc
    if count <= 0:
        raise CloudConstraintError("chat template produced no training tokens")
    return count


def preflight_training_text(
    *, tokenizer: object, dataset_root: Path, max_sequence_length: int
) -> TextPreflightReport:
    """Validate every chat row and token length before allocating model weights on GPU."""

    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise CloudConstraintError("tokenizer has no explicit chat template")
    if max_sequence_length <= 0:
        raise CloudConstraintError("maximum sequence length is invalid")
    surface_kinds = {
        "sft.jsonl": "sft",
        "sft_validation.jsonl": "sft",
        "preference.jsonl": "preference",
        "preference_validation.jsonl": "preference",
    }
    checked = 0
    maximum = 0
    for name, kind in surface_kinds.items():
        path = dataset_root / name
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise CloudConstraintError(f"training text surface is unreadable: {name}") from exc
        if not any(line.strip() for line in lines):
            raise CloudConstraintError(f"training text surface is empty: {name}")
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CloudConstraintError(
                    f"training text schema is invalid: {name}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise CloudConstraintError(f"training text schema is invalid: {name}:{line_number}")
            sequences = (
                (_message_list(row, "messages"),)
                if kind == "sft"
                else (
                    (*_message_list(row, "prompt"), *_message_list(row, "chosen")),
                    (*_message_list(row, "prompt"), *_message_list(row, "rejected")),
                )
            )
            counts = tuple(_token_count(tokenizer, sequence) for sequence in sequences)
            row_maximum = max(counts)
            if row_maximum > max_sequence_length:
                raise CloudConstraintError(
                    f"training text token length exceeds limit: {name}:{line_number}"
                )
            maximum = max(maximum, row_maximum)
            checked += 1
    return TextPreflightReport(checked, maximum, max_sequence_length)


def _build_gguf(
    *,
    plan: CloudExecutionPlan,
    source: Path,
    adapter: Path,
    artifacts: Path,
    cuda_enabled: bool = True,
) -> Path:  # pragma: no cover - requires the pinned native CUDA build image
    llama_cpp = plan.workspace / "llama.cpp"
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "https://github.com/ggml-org/llama.cpp.git",
            str(llama_cpp),
        ]
    )
    _checkout_pinned_revision(llama_cpp, plan.config.llama_cpp_revision)
    _run(
        [
            "cmake",
            "-B",
            "build",
            "-DLLAMA_CURL=OFF",
            f"-DGGML_CUDA={'ON' if cuda_enabled else 'OFF'}",
            "-DBUILD_SHARED_LIBS=OFF",
        ],
        cwd=llama_cpp,
    )
    _run(
        [
            "cmake",
            "--build",
            "build",
            "--config",
            "Release",
            "--target",
            "llama-server",
            "llama-export-lora",
            "llama-quantize",
            "-j1",
        ],
        cwd=llama_cpp,
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    benchmark_server = artifacts / "llama-server"
    shutil.copy2(llama_cpp / "build/bin/llama-server", benchmark_server)
    benchmark_server.chmod(0o755)
    base = artifacts / "base-f16.gguf"
    lora = artifacts / "adapter.gguf"
    merged = artifacts / "merged-f16.gguf"
    final = artifacts / "metaflora-incubus-v1.gguf"
    _run(
        [
            "python",
            str(llama_cpp / "convert_hf_to_gguf.py"),
            str(source),
            "--outfile",
            str(base),
            "--outtype",
            "f16",
        ]
    )
    _run(
        [
            "python",
            str(llama_cpp / "convert_lora_to_gguf.py"),
            "--base",
            str(source),
            "--outfile",
            str(lora),
            str(adapter),
        ]
    )
    if source.is_dir():
        shutil.rmtree(source)
    _run(
        [
            str(llama_cpp / "build/bin/llama-export-lora"),
            "-m",
            str(base),
            "-o",
            str(merged),
            "--lora",
            str(lora),
        ]
    )
    base.unlink(missing_ok=True)
    lora.unlink(missing_ok=True)
    try:
        _run(
            [
                str(llama_cpp / "build/bin/llama-quantize"),
                str(merged),
                str(final),
                plan.final_gguf_quantization,
            ]
        )
    finally:
        merged.unlink(missing_ok=True)
    if (
        not final.is_file()
        or not MIN_MODEL_BYTES <= final.stat().st_size <= plan.config.profile.final_gguf_max_bytes
    ):
        raise CloudConstraintError(
            f"{plan.final_gguf_quantization} GGUF must remain inside the 2.5-5 GiB cloud range"
        )
    return final


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reusable_final_gguf(*, plan: CloudExecutionPlan, checkpoint_root: Path) -> Path | None:
    """Return a complete private GGUF that can safely skip another export."""

    artifact = checkpoint_root / "artifacts" / "metaflora-incubus-v1.gguf"
    if not artifact.is_file():
        return None
    size = artifact.stat().st_size
    if not MIN_MODEL_BYTES <= size <= plan.config.profile.final_gguf_max_bytes:
        return None
    if not (artifact.parent / "llama-server").is_file():
        return None
    return artifact


def _write_artifact_state(*, checkpoint_root: Path, final: Path, phase: str) -> dict[str, object]:
    state = {
        "artifact_sha256": _sha256_file(final),
        "artifact_size_bytes": final.stat().st_size,
        "phase": phase,
        "schema_version": 1,
    }
    path = checkpoint_root / "artifacts" / "recovery-state.json"
    path.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return state


def _benchmark_final_gguf(
    *,
    plan: CloudExecutionPlan,
    artifact: Path,
    environment: Mapping[str, str],
    gpu_layers: int = 999,
) -> dict[str, object]:
    """Run the committed release cases before the cloud workspace can be removed."""
    from metaflora_incubus.gguf_benchmark_runner import (
        BenchmarkRunnerConfig,
        run_gguf_benchmark,
    )

    synced_server = artifact.parent / "llama-server"
    server = (
        synced_server
        if synced_server.is_file()
        else plan.workspace / "llama.cpp" / "build" / "bin" / "llama-server"
    )
    server.chmod(0o755)
    cases = Path(__file__).resolve().parents[2] / "benchmarks" / "gguf-v1-cases.jsonl"
    output = artifact.parent / "benchmark-final"
    config = BenchmarkRunnerConfig.create(
        server_binary=server,
        server_sha256=_sha256_file(server),
        model_path=artifact,
        model_sha256=_sha256_file(artifact),
        cases_path=cases,
        output_dir=output,
        seed=4242,
        port=18081,
        health_timeout_seconds=120.0,
        request_timeout_seconds=120.0,
        runner_code_revision=_required_revision(environment, "INCUBUS_CODE_REVISION"),
        gpu_layers=gpu_layers,
    )
    return run_gguf_benchmark(config)


def recover_trained_artifact(
    *,
    plan: CloudExecutionPlan,
    environment: Mapping[str, str],
    cpu_fallback: bool = False,
) -> dict[str, object]:  # pragma: no cover - requires the pinned native cloud image
    """Export and benchmark an authenticated final adapter without loading trainers."""

    from huggingface_hub import snapshot_download

    _require_cached_huggingface_auth()
    checkpoint_key = _required(environment, "INCUBUS_CHECKPOINT_HMAC_KEY")
    source_repo = _required(environment, "INCUBUS_SOURCE_REPO")
    source_revision = _required_revision(environment, "INCUBUS_SOURCE_REVISION")
    store = _checkpoint_store(plan)
    if plan.config.workspace.is_symlink() or plan.workspace.is_symlink():
        raise CloudConstraintError("refusing symlinked ephemeral workspace")
    plan.config.workspace.mkdir(parents=True, exist_ok=True)
    if plan.workspace.exists() or plan.workspace.is_symlink():
        safe_ephemeral_cleanup(plan)
    plan.workspace.mkdir(parents=False, exist_ok=False)
    checkpoint_root = plan.workspace / "checkpoints"
    restore = getattr(store, "restore_recovery", None) or store.restore
    restored = restore(checkpoint_root)
    if restored is None or not (checkpoint_root / "final-adapter").is_dir():
        raise CloudConstraintError("authenticated final adapter is missing")
    binding = _authenticated_recovery_binding(
        restored,
        environment=environment,
        checkpoint_key=checkpoint_key,
        run_id=plan.run_id,
    )
    for training_directory in (checkpoint_root / "sft", checkpoint_root / "preference"):
        if training_directory.is_dir():
            shutil.rmtree(training_directory)
    final = _reusable_final_gguf(plan=plan, checkpoint_root=checkpoint_root)
    if final is None:
        validate_cloud_disk_preflight(plan, available_bytes=_available_workspace_bytes(plan))
        source = plan.workspace / "source"
        snapshot_download(
            repo_id=source_repo,
            revision=source_revision,
            token=None,
            local_dir=source,
            allow_patterns=(
                "*.safetensors",
                "*.json",
                "tokenizer.model",
                "*.tiktoken",
                "vocab.*",
                "merges.txt",
            ),
        )
        with _hidden_huggingface_credentials(Path.home()):
            final = _build_gguf(
                plan=plan,
                source=source,
                adapter=checkpoint_root / "final-adapter",
                artifacts=checkpoint_root / "artifacts",
                cuda_enabled=not cpu_fallback,
            )
            _write_artifact_state(
                checkpoint_root=checkpoint_root,
                final=final,
                phase="artifact_built",
            )
        _write_checkpoint_manifest(checkpoint_root, binding=binding, key=checkpoint_key)
        store.sync(checkpoint_root)
    with _hidden_huggingface_credentials(Path.home()):
        benchmark = _benchmark_final_gguf(
            plan=plan,
            artifact=final,
            environment=environment,
            gpu_layers=0 if cpu_fallback else 999,
        )
    metadata = {
        "artifact_sha256": _sha256_file(final),
        "artifact_size_bytes": final.stat().st_size,
        "benchmark": benchmark,
        "gguf_quantization": plan.final_gguf_quantization,
        "schema_version": 1,
    }
    metadata_path = final.parent / "artifact-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _write_artifact_state(
        checkpoint_root=checkpoint_root,
        final=final,
        phase="benchmark_complete",
    )
    _write_checkpoint_manifest(checkpoint_root, binding=binding, key=checkpoint_key)
    store.sync(checkpoint_root)
    result = {
        **metadata,
        "checkpoint_remote": True,
        "product_id": plan.config.product_id,
        "public_upload": "blocked_until_eval_gates",
        "recovered_without_training": True,
    }
    safe_ephemeral_cleanup(plan)
    return result


def execute_training_and_build(
    *, plan: CloudExecutionPlan, environment: Mapping[str, str]
) -> dict[str, object]:  # pragma: no cover - exercised by the GPU notebook smoke job
    """Train and build privately; public upload remains impossible without eval authorization."""

    import torch
    from datasets import interleave_datasets, load_dataset
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        set_seed,
    )
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise CloudConstraintError("CUDA GPU is required")
    source_repo = _required(environment, "INCUBUS_SOURCE_REPO")
    dataset_repo = _required(environment, "INCUBUS_DATASET_REPO")
    source_revision = _required_revision(environment, "INCUBUS_SOURCE_REVISION")
    dataset_revision = _required_revision(environment, "INCUBUS_DATASET_REVISION")
    code_revision = _required_revision(environment, "INCUBUS_CODE_REVISION")
    dataset_sha = _required(environment, "INCUBUS_DATASET_SHA256")
    checkpoint_key = _required(environment, "INCUBUS_CHECKPOINT_HMAC_KEY")
    binding = {
        "code_revision": code_revision,
        "config_sha256": plan.config.config_sha256,
        "dataset_repo_sha256": hashlib.sha256(dataset_repo.encode()).hexdigest(),
        "dataset_revision": dataset_revision,
        "dataset_sha256": dataset_sha,
        "run_id": plan.run_id,
        "source_repo_sha256": hashlib.sha256(source_repo.encode()).hexdigest(),
        "source_revision": source_revision,
    }
    _require_cached_huggingface_auth()
    store = _checkpoint_store(plan)
    restored = _prepare_ephemeral_workspace(
        plan=plan,
        store=store,
        binding=binding,
        checkpoint_key=checkpoint_key,
        before_restore=lambda: validate_cloud_disk_preflight(
            plan, available_bytes=_available_workspace_bytes(plan)
        ),
    )
    source = plan.workspace / "source"
    data = plan.workspace / "data"
    snapshot_download(
        repo_id=source_repo,
        revision=source_revision,
        token=None,
        local_dir=source,
        allow_patterns=(
            "*.safetensors",
            "*.json",
            "tokenizer.model",
            "*.tiktoken",
            "vocab.*",
            "merges.txt",
        ),
    )
    snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        revision=dataset_revision,
        token=None,
        local_dir=data,
        allow_patterns=("manifest.json", "*.jsonl"),
    )

    from metaflora_incubus.training_entrypoints import _load_dataset_manifest

    _load_dataset_manifest(data / "manifest.json", dataset_sha)
    checkpoint_root = plan.workspace / "checkpoints"

    def sync_checkpoints() -> None:
        _write_checkpoint_manifest(checkpoint_root, binding=binding, key=checkpoint_key)
        store.sync(checkpoint_root)

    class RemoteSyncCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            del args, state, control, kwargs
            sync_checkpoints()

    set_seed(1701, deterministic=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    source_config = AutoConfig.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False
    )
    preflight_training_text(
        tokenizer=tokenizer,
        dataset_root=data,
        max_sequence_length=plan.config.profile.max_sequence_length,
    )
    architectures = tuple(getattr(source_config, "architectures", ()) or ())
    loader_kind = _select_model_loader_kind(architectures)
    model_loader = (
        AutoModelForImageTextToText if loader_kind == "image_text_to_text" else AutoModelForCausalLM
    )
    model = model_loader.from_pretrained(
        str(source),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float16,
        quantization_config=quantization,
        device_map={"": 0},
    )
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters > plan.config.profile.max_parameter_count:
        raise CloudConstraintError("loaded model exceeds the compact free-tier parameter limit")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    target_modules = _select_lora_targets(tuple(name for name, _ in model.named_modules()))
    lora = LoraConfig(
        r=plan.config.profile.lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    training_length = min(plan.config.profile.max_sequence_length, 1024)
    callback = RemoteSyncCallback()
    sft_output = checkpoint_root / "sft"
    sft_resume = _latest_checkpoint(restored / "sft") if restored else None
    preference_output = checkpoint_root / "preference"
    preference_resume = _latest_checkpoint(restored / "preference") if restored else None
    if preference_resume is not None:
        model = PeftModel.from_pretrained(model, str(preference_resume), is_trainable=True)
        sft = None
    else:
        sft = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=_mixed_dataset(
                load_dataset, interleave_datasets, data / "sft.jsonl", 1701
            ),
            eval_dataset=_mixed_dataset(
                load_dataset, interleave_datasets, data / "sft_validation.jsonl", 1701
            ),
            peft_config=lora,
            callbacks=[callback],
            args=SFTConfig(
                output_dir=str(sft_output),
                max_length=training_length,
                max_steps=32,
                save_steps=8,
                save_total_limit=2,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                fp16=True,
                gradient_checkpointing=True,
                report_to="none",
                seed=1701,
                data_seed=1701,
            ),
        )
        _cast_trainable_parameters_to_fp32(sft.model, torch_module=torch)
        sft.train(resume_from_checkpoint=str(sft_resume) if sft_resume else None)
        sft.save_model(str(sft_output / "final"))
        sync_checkpoints()
        model = sft.model
        sft = None
        gc.collect()
        torch.cuda.empty_cache()
    preference = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=_mixed_dataset(
            load_dataset, interleave_datasets, data / "preference.jsonl", 1702
        ),
        eval_dataset=_mixed_dataset(
            load_dataset,
            interleave_datasets,
            data / "preference_validation.jsonl",
            1702,
        ),
        callbacks=[callback],
        args=DPOConfig(
            output_dir=str(preference_output),
            max_length=training_length,
            max_steps=12,
            save_steps=3,
            save_total_limit=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-6,
            fp16=True,
            gradient_checkpointing=True,
            report_to="none",
            seed=1702,
            data_seed=1702,
        ),
    )
    _cast_trainable_parameters_to_fp32(preference.model, torch_module=torch)
    preference.train(resume_from_checkpoint=str(preference_resume) if preference_resume else None)
    adapter = checkpoint_root / "final-adapter"
    preference.save_model(str(adapter))
    sync_checkpoints()
    del model, sft, preference
    torch.cuda.empty_cache()
    with _hidden_huggingface_credentials(Path.home()):
        final = _build_gguf(
            plan=plan, source=source, adapter=adapter, artifacts=checkpoint_root / "artifacts"
        )
        benchmark = _benchmark_final_gguf(
            plan=plan,
            artifact=final,
            environment=environment,
        )
    artifact_sha256 = _sha256_file(final)
    metadata = {
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": final.stat().st_size,
        "benchmark": benchmark,
        "gguf_quantization": plan.final_gguf_quantization,
        "schema_version": 1,
    }
    metadata_path = final.parent / "artifact-metadata.json"
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    sync_checkpoints()
    result = {
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": final.stat().st_size,
        "benchmark": benchmark,
        "gguf_quantization": plan.final_gguf_quantization,
        "checkpoint_remote": True,
        "product_id": plan.config.product_id,
        "public_upload": "blocked_until_eval_gates",
    }
    safe_ephemeral_cleanup(plan)
    return result
