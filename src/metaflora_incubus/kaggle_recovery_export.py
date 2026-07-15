"""Crash-safe, resumable Kaggle export of a completed DPO adapter to Q5_K_M GGUF."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

GIB = 1024**3
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,63}")
_GGUF_MAGIC = b"GGUF"
_STATE_NAME = "recovery-state.json"
_MANIFEST_NAME = "candidate-export.json"
_SHA256_NAME = "candidate-sha256.txt"
_UPLOAD_RECEIPT_NAME = "candidate-upload-receipt.json"
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
_REVISION = re.compile(r"[0-9a-f]{40}")


class ExportRecoveryError(ValueError):
    """Raised when an export cannot resume safely."""


@dataclass(frozen=True)
class ResourceSnapshot:
    available_ram_bytes: int
    free_disk_bytes: int


@dataclass(frozen=True)
class RecoveryExportConfig:
    run_id: str
    base_model: Path
    adapter: Path
    workspace: Path
    merge_script: Path
    convert_script: Path
    quantize_binary: Path
    minimum_ram_bytes: int
    minimum_free_disk_bytes: int
    minimum_output_bytes: int
    maximum_output_bytes: int

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        base_model: Path | str,
        adapter: Path | str,
        workspace: Path | str,
        merge_script: Path | str,
        convert_script: Path | str,
        quantize_binary: Path | str,
        minimum_ram_bytes: int = 20 * GIB,
        minimum_free_disk_bytes: int = 24 * GIB,
        minimum_output_bytes: int = 5 * GIB // 2,
        maximum_output_bytes: int = 5 * GIB,
    ) -> RecoveryExportConfig:
        if _RUN_ID.fullmatch(run_id) is None:
            raise ExportRecoveryError("run identity is invalid")
        base = _required_directory(Path(base_model), "base model")
        adapter_path = _required_directory(Path(adapter), "DPO adapter")
        merger = _required_file(Path(merge_script), "merge helper")
        converter = _required_file(Path(convert_script), "GGUF converter")
        quantizer = _required_file(Path(quantize_binary), "GGUF quantizer")
        if (
            not (adapter_path / "adapter_config.json").is_file()
            or not (adapter_path / "adapter_model.safetensors").is_file()
        ):
            raise ExportRecoveryError("completed DPO adapter files are missing")
        workspace_path = Path(workspace)
        if workspace_path.is_symlink():
            raise ExportRecoveryError("recovery workspace is invalid")
        work = workspace_path.resolve()
        if work.exists() and not work.is_dir():
            raise ExportRecoveryError("recovery workspace is invalid")
        for value, label in (
            (minimum_ram_bytes, "minimum RAM"),
            (minimum_free_disk_bytes, "minimum free disk"),
            (minimum_output_bytes, "minimum output size"),
            (maximum_output_bytes, "maximum output size"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ExportRecoveryError(f"{label} is invalid")
        if minimum_output_bytes > maximum_output_bytes:
            raise ExportRecoveryError("output size range is invalid")
        return cls(
            run_id=run_id,
            base_model=base,
            adapter=adapter_path,
            workspace=work,
            merge_script=merger,
            convert_script=converter,
            quantize_binary=quantizer,
            minimum_ram_bytes=minimum_ram_bytes,
            minimum_free_disk_bytes=minimum_free_disk_bytes,
            minimum_output_bytes=minimum_output_bytes,
            maximum_output_bytes=maximum_output_bytes,
        )


@dataclass(frozen=True)
class RecoveryExportResult:
    artifact: Path
    artifact_sha256: str
    artifact_size_bytes: int
    manifest: Path
    resumed: bool


@dataclass(frozen=True)
class PrivateCandidateUploadResult:
    artifact_revision: str
    evidence_revision: str
    remote_prefix: str
    receipt: Path


class PrivateCandidateStore(Protocol):
    def ensure_private(self, *, repo_id: str, branch: str) -> None: ...

    def upload_candidate(
        self,
        *,
        repo_id: str,
        branch: str,
        folder: Path,
        remote_prefix: str,
        filenames: tuple[str, ...],
    ) -> str: ...

    def verify_candidate(
        self,
        *,
        repo_id: str,
        revision: str,
        remote_prefix: str,
        artifact_name: str,
        artifact_sha256: str,
        artifact_size_bytes: int,
        evidence_sha256: Mapping[str, str],
    ) -> bool: ...

    def upload_receipt(
        self,
        *,
        repo_id: str,
        branch: str,
        receipt: Path,
        path_in_repo: str,
        parent_revision: str,
    ) -> str: ...

    def verify_receipt(
        self,
        *,
        repo_id: str,
        revision: str,
        path_in_repo: str,
        expected_sha256: str,
    ) -> bool: ...


ResourceProbe = Callable[[Path], ResourceSnapshot]
CommandRunner = Callable[[tuple[str, ...]], None]


def execute_recovery_export(
    config: RecoveryExportConfig,
    *,
    resources: ResourceProbe | None = None,
    command_runner: CommandRunner | None = None,
) -> RecoveryExportResult:
    """Resume merge/convert/quantize stages and return an artifact-bound receipt."""

    probe = resources or probe_resources
    run_command = command_runner or _run_command
    config.workspace.mkdir(parents=True, mode=0o700, exist_ok=True)
    if config.workspace.is_symlink():
        raise ExportRecoveryError("recovery workspace is invalid")
    config.workspace.chmod(0o700)
    inputs = {
        "adapter_sha256": _sha256_directory(config.adapter),
        "base_model_sha256": _sha256_directory(config.base_model),
    }
    state_path = config.workspace / _STATE_NAME
    state, resumed = _load_state(state_path, run_id=config.run_id, inputs=inputs)
    stages = dict(state["stages"])
    merged = config.workspace / "merged-safetensors"
    f16 = config.workspace / "incubus-v1-f16.gguf"
    artifact = config.workspace / "metaflora-incubus-v1.gguf"

    if _valid_file_stage(
        artifact,
        stages.get("quantize"),
        minimum_size=config.minimum_output_bytes,
        maximum_size=config.maximum_output_bytes,
    ):
        stages = _record_file_stage(stages, "quantize", artifact)
        state = _write_state(state_path, config.run_id, inputs, stages)
        return _write_result(config, artifact, state, resumed=True)
    _discard_invalid_file(artifact, config.workspace / f"{artifact.name}.partial")
    stages.pop("quantize", None)

    if not _valid_file_stage(f16, stages.get("convert"), minimum_size=4):
        _discard_invalid_file(f16, config.workspace / f"{f16.name}.partial")
        stages.pop("convert", None)
        if not _valid_directory_stage(merged, stages.get("merge")):
            _discard_invalid_directory(merged, config.workspace / f"{merged.name}.partial")
            stages.pop("merge", None)
            _require_resources(
                config,
                probe(config.workspace),
                stage="merge",
                required_disk_bytes=max(
                    config.minimum_free_disk_bytes,
                    _directory_size(config.base_model) + config.maximum_output_bytes,
                ),
            )
            partial_merged = config.workspace / f"{merged.name}.partial"
            run_command(
                (
                    sys.executable,
                    str(config.merge_script),
                    "--base",
                    str(config.base_model),
                    "--adapter",
                    str(config.adapter),
                    "--output",
                    str(partial_merged),
                )
            )
            _require_merged_directory(partial_merged)
            partial_merged.replace(merged)
            stages = _record_directory_stage(stages, "merge", merged)
            state = _write_state(state_path, config.run_id, inputs, stages)
        else:
            stages = _record_directory_stage(stages, "merge", merged)
            state = _write_state(state_path, config.run_id, inputs, stages)

        _require_resources(
            config,
            probe(config.workspace),
            stage="convert",
            required_disk_bytes=_directory_size(merged) + config.maximum_output_bytes,
        )
        partial_f16 = config.workspace / f"{f16.name}.partial"
        run_command(
            (
                sys.executable,
                str(config.convert_script),
                str(merged),
                "--outfile",
                str(partial_f16),
                "--outtype",
                "f16",
            )
        )
        _require_gguf(partial_f16, minimum_size=4)
        partial_f16.replace(f16)
        f16.chmod(0o600)
        stages = _record_file_stage(stages, "convert", f16)
        state = _write_state(state_path, config.run_id, inputs, stages)
    else:
        stages = _record_file_stage(stages, "convert", f16)
        state = _write_state(state_path, config.run_id, inputs, stages)

    _require_resources(
        config,
        probe(config.workspace),
        stage="quantize",
        required_disk_bytes=config.maximum_output_bytes + GIB,
    )
    partial_artifact = config.workspace / f"{artifact.name}.partial"
    partial_artifact.unlink(missing_ok=True)
    run_command((str(config.quantize_binary), str(f16), str(partial_artifact), "Q5_K_M"))
    _require_gguf(
        partial_artifact,
        minimum_size=config.minimum_output_bytes,
        maximum_size=config.maximum_output_bytes,
    )
    partial_artifact.replace(artifact)
    artifact.chmod(0o600)
    stages = _record_file_stage(stages, "quantize", artifact)
    state = _write_state(state_path, config.run_id, inputs, stages)
    return _write_result(config, artifact, state, resumed=resumed)


def upload_private_candidate(
    *,
    config: RecoveryExportConfig,
    result: RecoveryExportResult,
    repo_id: str,
    branch: str,
    store: PrivateCandidateStore | None = None,
) -> PrivateCandidateUploadResult:
    """Persist one completed candidate to a private Hub revision before Kaggle exits."""

    if _REPO_ID.fullmatch(repo_id) is None:
        raise ExportRecoveryError("private checkpoint repository is invalid")
    if (
        _BRANCH.fullmatch(branch) is None
        or ".." in branch
        or branch.endswith("/")
        or "//" in branch
    ):
        raise ExportRecoveryError("private checkpoint branch is invalid")
    if result.artifact.parent != config.workspace or result.manifest.parent != config.workspace:
        raise ExportRecoveryError("candidate upload paths are outside the recovery workspace")
    _require_gguf(
        result.artifact,
        minimum_size=config.minimum_output_bytes,
        maximum_size=config.maximum_output_bytes,
    )
    if _sha256_file(result.artifact) != result.artifact_sha256:
        raise ExportRecoveryError("candidate changed before private upload")
    if result.artifact.stat().st_size != result.artifact_size_bytes:
        raise ExportRecoveryError("candidate size changed before private upload")
    if not result.manifest.is_file() or result.manifest.is_symlink():
        raise ExportRecoveryError("candidate manifest is invalid")

    checksum = config.workspace / _SHA256_NAME
    _atomic_text(checksum, f"{result.artifact_sha256}  {result.artifact.name}\n")
    filenames = tuple(sorted((result.manifest.name, checksum.name, result.artifact.name)))
    evidence_sha256 = {
        result.manifest.name: _sha256_file(result.manifest),
        checksum.name: _sha256_file(checksum),
    }
    remote_prefix = f"runs/{config.run_id}/exports/q5-k-m/{result.artifact_sha256}"
    receipt_path_in_repo = f"runs/{config.run_id}/exports/{_UPLOAD_RECEIPT_NAME}"
    private_store = store or HuggingFacePrivateCandidateStore()
    private_store.ensure_private(repo_id=repo_id, branch=branch)
    artifact_revision = private_store.upload_candidate(
        repo_id=repo_id,
        branch=branch,
        folder=config.workspace,
        remote_prefix=remote_prefix,
        filenames=filenames,
    )
    if _REVISION.fullmatch(artifact_revision) is None:
        raise ExportRecoveryError("private candidate upload returned an invalid revision")
    if not private_store.verify_candidate(
        repo_id=repo_id,
        revision=artifact_revision,
        remote_prefix=remote_prefix,
        artifact_name=result.artifact.name,
        artifact_sha256=result.artifact_sha256,
        artifact_size_bytes=result.artifact_size_bytes,
        evidence_sha256=evidence_sha256,
    ):
        raise ExportRecoveryError("private candidate upload verification failed")

    receipt = config.workspace / _UPLOAD_RECEIPT_NAME
    _atomic_json(
        receipt,
        {
            "artifact_path": f"{remote_prefix}/{result.artifact.name}",
            "artifact_revision": artifact_revision,
            "artifact_sha256": result.artifact_sha256,
            "artifact_size_bytes": result.artifact_size_bytes,
            "branch": branch,
            "gguf_quantization": "Q5_K_M",
            "release_ready": False,
            "remote_prefix": remote_prefix,
            "repo_id": repo_id,
            "required_next_step": "run_parity_and_release_gates",
            "run_id": config.run_id,
            "schema_version": 1,
        },
        pretty=True,
    )
    evidence_revision = private_store.upload_receipt(
        repo_id=repo_id,
        branch=branch,
        receipt=receipt,
        path_in_repo=receipt_path_in_repo,
        parent_revision=artifact_revision,
    )
    if _REVISION.fullmatch(evidence_revision) is None:
        raise ExportRecoveryError("private upload receipt returned an invalid revision")
    if not private_store.verify_receipt(
        repo_id=repo_id,
        revision=evidence_revision,
        path_in_repo=receipt_path_in_repo,
        expected_sha256=_sha256_file(receipt),
    ):
        raise ExportRecoveryError("private upload receipt verification failed")
    return PrivateCandidateUploadResult(
        artifact_revision=artifact_revision,
        evidence_revision=evidence_revision,
        remote_prefix=remote_prefix,
        receipt=receipt,
    )


class HuggingFacePrivateCandidateStore:
    """Minimal authenticated Hub adapter for content-addressed recovery artifacts."""

    def __init__(self) -> None:
        from huggingface_hub import HfApi

        self._api = HfApi(token=None)

    def ensure_private(self, *, repo_id: str, branch: str) -> None:
        info = self._api.model_info(repo_id=repo_id)
        if info.private is not True:
            raise ExportRecoveryError("checkpoint repository is not private")
        self._api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)

    def upload_candidate(
        self,
        *,
        repo_id: str,
        branch: str,
        folder: Path,
        remote_prefix: str,
        filenames: tuple[str, ...],
    ) -> str:
        commit = self._api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            folder_path=str(folder),
            path_in_repo=remote_prefix,
            allow_patterns=list(filenames),
            commit_message="Store private Q5_K_M recovery candidate",
        )
        return str(commit.oid).lower()

    def verify_candidate(
        self,
        *,
        repo_id: str,
        revision: str,
        remote_prefix: str,
        artifact_name: str,
        artifact_sha256: str,
        artifact_size_bytes: int,
        evidence_sha256: Mapping[str, str],
    ) -> bool:
        from huggingface_hub import hf_hub_download

        info = self._api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
        if info.private is not True or str(info.sha).lower() != revision:
            return False
        siblings = {item.rfilename: item for item in info.siblings}
        artifact_path = f"{remote_prefix}/{artifact_name}"
        artifact = siblings.get(artifact_path)
        if (
            artifact is None
            or artifact.size != artifact_size_bytes
            or artifact.lfs is None
            or artifact.lfs.sha256 != artifact_sha256
            or artifact.lfs.size != artifact_size_bytes
        ):
            return False
        for filename, expected_sha256 in evidence_sha256.items():
            remote_path = f"{remote_prefix}/{filename}"
            if remote_path not in siblings:
                return False
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="model",
                    revision=revision,
                    filename=remote_path,
                    token=None,
                )
            )
            if _sha256_file(downloaded) != expected_sha256:
                return False
        return True

    def upload_receipt(
        self,
        *,
        repo_id: str,
        branch: str,
        receipt: Path,
        path_in_repo: str,
        parent_revision: str,
    ) -> str:
        commit = self._api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            path_or_fileobj=str(receipt),
            path_in_repo=path_in_repo,
            parent_commit=parent_revision,
            commit_message="Record immutable recovery candidate revision",
        )
        return str(commit.oid).lower()

    def verify_receipt(
        self,
        *,
        repo_id: str,
        revision: str,
        path_in_repo: str,
        expected_sha256: str,
    ) -> bool:
        from huggingface_hub import hf_hub_download

        info = self._api.model_info(repo_id=repo_id, revision=revision)
        if info.private is not True or str(info.sha).lower() != revision:
            return False
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                filename=path_in_repo,
                token=None,
            )
        )
        return _sha256_file(downloaded) == expected_sha256


def probe_resources(path: Path) -> ResourceSnapshot:
    """Read currently available RAM and free bytes on the workspace filesystem."""

    available_ram = _available_ram_bytes()
    free_disk = shutil.disk_usage(path).free
    return ResourceSnapshot(available_ram_bytes=available_ram, free_disk_bytes=free_disk)


def _available_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(page_size) * int(available_pages)


def _require_resources(
    config: RecoveryExportConfig,
    snapshot: ResourceSnapshot,
    *,
    stage: str,
    required_disk_bytes: int,
) -> None:
    if stage == "merge" and snapshot.available_ram_bytes < config.minimum_ram_bytes:
        raise ExportRecoveryError(
            f"insufficient available RAM for merge: {snapshot.available_ram_bytes} bytes"
        )
    if snapshot.free_disk_bytes < required_disk_bytes:
        raise ExportRecoveryError(
            f"insufficient free disk for {stage}: need {required_disk_bytes} bytes, "
            f"have {snapshot.free_disk_bytes} bytes"
        )


def _load_state(
    path: Path, *, run_id: str, inputs: Mapping[str, str]
) -> tuple[dict[str, object], bool]:
    if not path.exists():
        return {"inputs": dict(inputs), "run_id": run_id, "schema_version": 1, "stages": {}}, False
    if path.is_symlink() or not path.is_file():
        raise ExportRecoveryError("recovery state is invalid")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportRecoveryError("recovery state is unreadable") from exc
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ExportRecoveryError("recovery state schema is invalid")
    if state.get("run_id") != run_id:
        raise ExportRecoveryError("recovery state run identity does not match")
    if state.get("inputs") != dict(inputs) or not isinstance(state.get("stages"), dict):
        raise ExportRecoveryError("recovery state input binding does not match")
    return state, True


def _write_state(
    path: Path, run_id: str, inputs: Mapping[str, str], stages: Mapping[str, object]
) -> dict[str, object]:
    state = {
        "inputs": dict(inputs),
        "run_id": run_id,
        "schema_version": 1,
        "stages": dict(stages),
    }
    _atomic_json(path, state)
    return state


def _write_result(
    config: RecoveryExportConfig,
    artifact: Path,
    state: Mapping[str, object],
    *,
    resumed: bool,
) -> RecoveryExportResult:
    artifact_sha = _sha256_file(artifact)
    stages = state["stages"]
    if not isinstance(stages, Mapping):
        raise ExportRecoveryError("recovery stages are invalid")
    manifest = {
        "artifact": {
            "path": artifact.name,
            "sha256": artifact_sha,
            "size_bytes": artifact.stat().st_size,
        },
        "candidate_state": "quantized_candidate",
        "gguf_quantization": "Q5_K_M",
        "release_ready": False,
        "required_next_step": "run_parity_and_release_gates",
        "resume": {
            "completed_stages": [
                name for name in ("merge", "convert", "quantize") if name in stages
            ]
        },
        "run_id": config.run_id,
        "schema_version": 1,
    }
    manifest_path = config.workspace / _MANIFEST_NAME
    _atomic_json(manifest_path, manifest, pretty=True)
    return RecoveryExportResult(
        artifact=artifact,
        artifact_sha256=artifact_sha,
        artifact_size_bytes=artifact.stat().st_size,
        manifest=manifest_path,
        resumed=resumed,
    )


def _record_file_stage(stages: Mapping[str, object], name: str, path: Path) -> dict[str, object]:
    return {
        **stages,
        name: {"path": path.name, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size},
    }


def _record_directory_stage(
    stages: Mapping[str, object], name: str, path: Path
) -> dict[str, object]:
    return {**stages, name: {"path": path.name, "sha256": _sha256_directory(path)}}


def _valid_file_stage(
    path: Path,
    record: object,
    *,
    minimum_size: int,
    maximum_size: int | None = None,
) -> bool:
    try:
        _require_gguf(path, minimum_size=minimum_size, maximum_size=maximum_size)
    except ExportRecoveryError:
        return False
    return (
        isinstance(record, Mapping)
        and record.get("path") == path.name
        and record.get("size_bytes") == path.stat().st_size
        and record.get("sha256") == _sha256_file(path)
    )


def _valid_directory_stage(path: Path, record: object) -> bool:
    try:
        _require_merged_directory(path)
    except ExportRecoveryError:
        return False
    return (
        isinstance(record, Mapping)
        and record.get("path") == path.name
        and record.get("sha256") == _sha256_directory(path)
    )


def _require_gguf(path: Path, *, minimum_size: int, maximum_size: int | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        raise ExportRecoveryError("GGUF export is missing")
    size = path.stat().st_size
    if size < minimum_size or (maximum_size is not None and size > maximum_size):
        raise ExportRecoveryError("GGUF export size is outside the allowed range")
    with path.open("rb") as handle:
        if handle.read(4) != _GGUF_MAGIC:
            raise ExportRecoveryError("GGUF export magic is invalid")


def _require_merged_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ExportRecoveryError("merged model directory is missing")
    if not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
        raise ExportRecoveryError("merged model directory is incomplete")


def _discard_invalid_file(path: Path, partial: Path) -> None:
    path.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)


def _discard_invalid_directory(path: Path, partial: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    if partial.exists():
        shutil.rmtree(partial)


def _required_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_dir():
        raise ExportRecoveryError(f"{label} directory is invalid")
    return resolved


def _required_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ExportRecoveryError(f"{label} is invalid")
    return resolved


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ExportRecoveryError(f"directory has no files: {path.name}")
    for item in files:
        if item.is_symlink():
            raise ExportRecoveryError(f"directory contains a symlink: {path.name}")
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object], *, pretty: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    path.chmod(0o600)


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    path.chmod(0o600)


def _run_command(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True, shell=False)
