"""Fail-closed execution contracts for free-tier cloud GPU training."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from metaflora_incubus.huggingface_publication import (
    MODEL_NAME,
    PublicationDecision,
    PublicationPolicy,
    PublicationResult,
    Uploader,
    publish_to_huggingface,
)

GIB = 1024**3
THREE_GIB = 3 * GIB
FIVE_GIB = 5 * GIB
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


class CloudConstraintError(ValueError):
    """Raised when a free-tier run would be unsafe or materially unrealistic."""


class CheckpointBackend(str, Enum):
    GOOGLE_DRIVE = "google_drive"
    HF_PRIVATE_BRANCH = "hf_private_branch"


@dataclass(frozen=True)
class FreeGpuProfile:
    max_parameter_count: int
    min_vram_bytes: int
    max_sequence_length: int
    lora_rank: int
    load_in_4bit: bool
    quantization_type: str
    final_gguf_max_bytes: int

    @classmethod
    def default(cls) -> FreeGpuProfile:
        return cls(
            max_parameter_count=7_500_000_000,
            min_vram_bytes=15 * GIB,
            max_sequence_length=2048,
            lora_rank=16,
            load_in_4bit=True,
            quantization_type="nf4",
            final_gguf_max_bytes=FIVE_GIB,
        )

    def validate(self, *, parameter_count: int, vram_bytes: int) -> None:
        if parameter_count <= 0 or parameter_count > self.max_parameter_count:
            raise CloudConstraintError(
                "parameter count exceeds the honest free-tier compact profile"
            )
        if vram_bytes < self.min_vram_bytes:
            raise CloudConstraintError("GPU VRAM is below the 15 GiB free-tier floor")


@dataclass(frozen=True)
class CloudConfig:
    product_id: str
    workspace: Path
    public_repo_id: str
    profile: FreeGpuProfile
    llama_cpp_revision: str
    config_sha256: str


@dataclass(frozen=True)
class RemoteCheckpointTarget:
    backend: CheckpointBackend
    location: str
    branch: str | None

    @classmethod
    def create(
        cls,
        *,
        backend: CheckpointBackend,
        location: str,
        branch: str | None,
    ) -> RemoteCheckpointTarget:
        value = location.strip()
        if backend is CheckpointBackend.GOOGLE_DRIVE:
            if not value.startswith("/content/drive/MyDrive/"):
                raise CloudConstraintError("Google Drive checkpoints must live under MyDrive")
            if branch is not None:
                raise CloudConstraintError("Google Drive checkpoints do not use a branch")
        elif backend is CheckpointBackend.HF_PRIVATE_BRANCH:
            if "/" not in value or value == "metaflora/incubus":
                raise CloudConstraintError(
                    "checkpoint repository must be a separate private branch"
                )
            if not branch or branch.strip() in {"main", "master"}:
                raise CloudConstraintError("checkpoint branch cannot be main or master")
            if not re.fullmatch(r"[A-Za-z0-9._/-]+", branch):
                raise CloudConstraintError("invalid checkpoint branch")
        else:
            raise CloudConstraintError("unsupported checkpoint backend")
        return cls(backend, value, branch.strip() if branch else None)


@dataclass(frozen=True)
class CloudExecutionPlan:
    config: CloudConfig
    checkpoint_target: RemoteCheckpointTarget
    run_id: str
    parameter_count: int
    workspace: Path
    local_retention: bool
    resume_enabled: bool
    training_mode: str
    post_training_steps: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        config: CloudConfig,
        checkpoint_target: RemoteCheckpointTarget,
        run_id: str,
        parameter_count: int,
        vram_bytes: int,
    ) -> CloudExecutionPlan:
        if not _RUN_ID.fullmatch(run_id):
            raise CloudConstraintError("invalid run_id")
        config.profile.validate(parameter_count=parameter_count, vram_bytes=vram_bytes)
        return cls(
            config=config,
            checkpoint_target=checkpoint_target,
            run_id=run_id,
            parameter_count=parameter_count,
            workspace=config.workspace / run_id,
            local_retention=False,
            resume_enabled=True,
            training_mode="qlora_nf4",
            post_training_steps=(
                "convert_base_to_f16_gguf",
                "convert_adapter_to_gguf",
                "merge_gguf_in_cloud",
                "quantize_q4_k_m",
                "run_eval_gates",
                "publish_verified_bundle",
                "delete_ephemeral_workspace",
            ),
        )


@dataclass(frozen=True)
class CloudDiskBudget:
    source_download_bytes: int
    build_intermediate_bytes: int
    final_artifact_bytes: int
    reserve_bytes: int
    required_bytes: int


def cloud_disk_budget(plan: CloudExecutionPlan) -> CloudDiskBudget:
    """Conservative peak budget before any remote bytes are downloaded."""

    source_download = plan.parameter_count * 2
    build_intermediates = plan.parameter_count * 4
    final_artifact = plan.config.profile.final_gguf_max_bytes
    reserve = 8 * GIB
    required = source_download + build_intermediates + final_artifact + reserve
    return CloudDiskBudget(
        source_download_bytes=source_download,
        build_intermediate_bytes=build_intermediates,
        final_artifact_bytes=final_artifact,
        reserve_bytes=reserve,
        required_bytes=required,
    )


def validate_cloud_disk_preflight(
    plan: CloudExecutionPlan, *, available_bytes: int
) -> CloudDiskBudget:
    budget = cloud_disk_budget(plan)
    if available_bytes < 0 or available_bytes < budget.required_bytes:
        raise CloudConstraintError(
            "disk preflight failed before download: "
            f"need {budget.required_bytes} bytes, found {available_bytes} bytes"
        )
    return budget


@dataclass(frozen=True)
class PublicationAuthorization:
    allowed: bool
    repo_id: str
    artifact_path: Path
    artifact_size_bytes: int
    artifact_sha256: str


def _canonical_json(document: object) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def load_cloud_config(path: Path | str) -> CloudConfig:
    config_path = Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloudConstraintError("invalid cloud config") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise CloudConstraintError("unsupported cloud config schema")
    profile = document.get("profile")
    if not isinstance(profile, dict):
        raise CloudConstraintError("cloud profile is required")
    result = CloudConfig(
        product_id=str(document.get("product_id", "")),
        workspace=Path(str(document.get("workspace", ""))),
        public_repo_id=str(document.get("public_repo_id", "")),
        profile=FreeGpuProfile(
            max_parameter_count=int(profile.get("max_parameter_count", 0)),
            min_vram_bytes=int(profile.get("min_vram_bytes", 0)),
            max_sequence_length=int(profile.get("max_sequence_length", 0)),
            lora_rank=int(profile.get("lora_rank", 0)),
            load_in_4bit=profile.get("load_in_4bit") is True,
            quantization_type=str(profile.get("quantization_type", "")),
            final_gguf_max_bytes=int(profile.get("final_gguf_max_bytes", 0)),
        ),
        llama_cpp_revision=str(document.get("llama_cpp_revision", "")),
        config_sha256=hashlib.sha256(_canonical_json(document)).hexdigest(),
    )
    if result.product_id != "metaflora-incubus-v1":
        raise CloudConstraintError("invalid product id")
    if result.workspace != Path("/content/incubus-work"):
        raise CloudConstraintError("workspace must remain on ephemeral cloud storage")
    if result.public_repo_id != "metaflora/incubus":
        raise CloudConstraintError("unexpected public repository")
    if result.profile != FreeGpuProfile.default():
        raise CloudConstraintError("free-tier profile drift")
    if not re.fullmatch(r"[0-9a-f]{40}", result.llama_cpp_revision):
        raise CloudConstraintError("llama.cpp revision must be a pinned 40-hex commit")
    return result


def authorize_publication(
    *, artifact_path: Path, gate_decision: PublicationDecision, repo_id: str
) -> PublicationAuthorization:
    """Authorize the public upload only after exact artifact and eval checks pass."""

    if gate_decision.approved is not True or gate_decision.blockers:
        raise CloudConstraintError("eval gates did not approve the deployable artifact")
    if repo_id != "metaflora/incubus":
        raise CloudConstraintError("public upload destination is not approved")
    if not artifact_path.is_file() or artifact_path.suffix.lower() != ".gguf":
        raise CloudConstraintError("deployable GGUF artifact is missing")
    size = artifact_path.stat().st_size
    if size < THREE_GIB:
        raise CloudConstraintError("final GGUF must be at least 3 GiB")
    if size > FIVE_GIB:
        raise CloudConstraintError("final GGUF must be greater than zero and at most 5 GiB")
    digest = hashlib.sha256()
    with artifact_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return PublicationAuthorization(True, repo_id, artifact_path, size, digest.hexdigest())


def publish_after_eval_gates(
    *,
    bundle: Path,
    artifact_path: Path,
    gate_decision: PublicationDecision,
    signature_verifier: Callable[[str, bytes, bytes], bool],
    uploader: Uploader,
    prohibited_identifiers: tuple[str, ...],
) -> PublicationResult:
    """Direct-upload a signed bundle only after both cloud and bundle gates pass."""

    authorization = authorize_publication(
        artifact_path=artifact_path,
        gate_decision=gate_decision,
        repo_id="metaflora/incubus",
    )
    expected_artifact = (bundle / MODEL_NAME).resolve()
    if authorization.artifact_path.resolve() != expected_artifact:
        raise CloudConstraintError("authorized artifact is not the bundle deployable")
    policy = PublicationPolicy(
        repo_id=authorization.repo_id,
        min_model_bytes=THREE_GIB,
        max_model_bytes=FIVE_GIB,
        prohibited_identifiers=prohibited_identifiers,
    )
    return publish_to_huggingface(
        bundle,
        policy=policy,
        signature_verifier=signature_verifier,
        uploader=uploader,
    )


class GoogleDriveCheckpointStore:
    """Byte-copy checkpoints to mounted Drive; no intermediate is sent to the Mac."""

    def __init__(self, target: RemoteCheckpointTarget, run_id: str) -> None:
        if target.backend is not CheckpointBackend.GOOGLE_DRIVE:
            raise CloudConstraintError("wrong checkpoint backend")
        self._root = Path(target.location) / run_id

    def restore(self, destination: Path) -> Path | None:
        if not self._root.is_dir():
            return None
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self._root, destination)
        return destination

    def sync(self, source: Path) -> None:
        temporary = self._root.with_name(f".{self._root.name}.uploading")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, temporary)
        if self._root.exists():
            shutil.rmtree(self._root)
        temporary.replace(self._root)


class HuggingFacePrivateCheckpointStore:
    """Resume checkpoints through a separate private Hub repository branch."""

    def __init__(self, target: RemoteCheckpointTarget, run_id: str, *, token: str) -> None:
        if target.backend is not CheckpointBackend.HF_PRIVATE_BRANCH or not token:
            raise CloudConstraintError("private Hub checkpoint credentials are required")
        from huggingface_hub import HfApi

        self._api = HfApi(token=token)
        self._target = target
        self._run_id = run_id
        self._token = token

    def ensure_private(self) -> None:
        self._api.create_repo(
            repo_id=self._target.location,
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        info = self._api.model_info(repo_id=self._target.location)
        if info.private is not True:
            raise CloudConstraintError("checkpoint repository is not private")
        self._api.create_branch(
            repo_id=self._target.location,
            branch=self._target.branch,
            exist_ok=True,
        )

    def restore(self, destination: Path) -> Path | None:
        from huggingface_hub import snapshot_download

        self.ensure_private()
        staging = destination.with_name(f".{destination.name}.restoring")
        if staging.exists():
            shutil.rmtree(staging)
        try:
            snapshot_download(
                repo_id=self._target.location,
                revision=self._target.branch,
                allow_patterns=f"runs/{self._run_id}/**",
                local_dir=staging,
                token=self._token,
            )
        except Exception as exc:
            if staging.exists():
                shutil.rmtree(staging)
            if exc.__class__.__name__ in {"EntryNotFoundError", "RevisionNotFoundError"}:
                return None
            raise
        restored = staging / "runs" / self._run_id
        if not restored.is_dir():
            shutil.rmtree(staging)
            return None
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(restored, destination)
        shutil.rmtree(staging)
        return destination

    def sync(self, source: Path) -> None:
        self.ensure_private()
        self._api.upload_folder(
            repo_id=self._target.location,
            repo_type="model",
            revision=self._target.branch,
            folder_path=str(source),
            path_in_repo=f"runs/{self._run_id}",
            commit_message=f"Checkpoint {self._run_id}",
        )


def safe_ephemeral_cleanup(plan: CloudExecutionPlan) -> None:
    if plan.config.workspace.is_symlink() or plan.workspace.is_symlink():
        raise CloudConstraintError("refusing symlinked workspace cleanup")
    if plan.workspace != plan.config.workspace / plan.run_id:
        raise CloudConstraintError("refusing cleanup outside the run workspace")
    root = plan.config.workspace.resolve()
    target = plan.workspace.resolve()
    if root not in target.parents or target == root:
        raise CloudConstraintError("refusing unsafe workspace cleanup")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def checkpoint_relative_path(run_id: str) -> PurePosixPath:
    if not _RUN_ID.fullmatch(run_id):
        raise CloudConstraintError("invalid run_id")
    return PurePosixPath("runs") / run_id
