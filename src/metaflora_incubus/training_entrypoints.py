"""Maintainer-only, fail-closed entrypoints for reproducible candidate training."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from metaflora_incubus.training_contract import (
    CheckpointCompatibilityError,
    CheckpointRef,
    ContaminationError,
    DataMix,
    DatasetCatalog,
    DatasetPartition,
    DatasetRecord,
    DatasetSplit,
    DuplicateExampleError,
    LicensePolicy,
    StageKind,
    TrainingConfig,
    TrainingPlan,
    TrainingStage,
    create_resume_plan,
    scan_contamination,
)

_SHA256_LENGTH = 64
_TRAINING_FILES = (
    "sft.jsonl",
    "preference.jsonl",
    "sft_validation.jsonl",
    "preference_validation.jsonl",
)
_PROVENANCE_FILE = "provenance.jsonl"
_MANIFEST_FILES = (*_TRAINING_FILES, _PROVENANCE_FILE)
_CAPABILITIES = ("code", "agentic_tools", "russian_text", "english_text")


class TrainingInputError(ValueError):
    """Raised when an input is unsafe, incomplete, or unsuitable for training."""


class HashMismatchError(TrainingInputError):
    """Raised when pinned provenance does not match bytes on disk."""


class MissingTrainingDependencyError(RuntimeError):
    """Raised only for a real run when optional ML dependencies are absent."""


class CandidateState(str, Enum):
    ADAPTER_ONLY = "adapter_only"


@dataclass(frozen=True)
class EnvironmentReference:
    path_env: str
    sha256_env: str


@dataclass(frozen=True)
class DatasetReference:
    manifest_env: str
    sha256_env: str


@dataclass(frozen=True)
class LoraSettings:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class MaintainerConfig:
    product_id: str
    document_sha256: str
    config_sha256_env: str
    source: EnvironmentReference
    dataset: DatasetReference
    output_dir: str
    seed: int
    sequence_length: int
    effective_batch_size: int
    per_device_train_batch_size: int
    learning_rate: float
    epochs: int
    precision: str
    license_policy: LicensePolicy
    lora: LoraSettings
    stages: tuple[TrainingStage, ...]


@dataclass(frozen=True)
class PreparedDataset:
    manifest_path: Path
    dataset_sha256: str
    input_sha256: str
    record_counts: Mapping[str, int]


@dataclass(frozen=True)
class StageRecipe:
    kind: StageKind
    trainer: str
    seed: int
    dataset_path: Path
    validation_path: Path
    output_dir: Path
    max_steps: int
    checkpoint_every: int
    data_mix: DataMix


@dataclass(frozen=True)
class PostTrainingRecipe:
    input_state: CandidateState
    steps: tuple[str, ...]


@dataclass(frozen=True)
class TrainingRun:
    product_id: str
    dry_run: bool
    plan: TrainingPlan
    source_path: Path
    dataset_manifest_path: Path
    output_dir: Path
    execution_config_sha256: str
    execution_plan_sha256: str
    model_load_kwargs: Mapping[str, bool]
    stage_recipes: tuple[StageRecipe, ...]
    resume_checkpoint: object | None
    resume_path: Path | None
    maintainer_config: MaintainerConfig
    candidate_state: CandidateState
    release_ready: bool
    post_training: PostTrainingRecipe


def _canonical_json(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_LENGTH or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise TrainingInputError(f"invalid {label} sha256")
    return normalized


def _require_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TrainingInputError(f"{label} must be an object")
    return value


def _require_string(document: Mapping[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TrainingInputError(f"invalid {key}")
    return value.strip()


def _read_json_object(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingInputError(f"cannot read JSON object: {path}") from exc
    mapping = _require_mapping(document, label=str(path))
    return mapping, _canonical_json(mapping)


def load_maintainer_config(path: Path | str) -> MaintainerConfig:
    """Load and validate the generic, brand-only maintainer configuration."""

    config_path = Path(path)
    document, canonical = _read_json_object(config_path)
    if document.get("schema_version") != 1:
        raise TrainingInputError("unsupported training config schema")

    source = _require_mapping(document.get("source"), label="source")
    dataset = _require_mapping(document.get("dataset"), label="dataset")
    lora = _require_mapping(document.get("lora"), label="lora")
    licenses = _require_mapping(document.get("licenses"), label="licenses")
    raw_stages = document.get("stages")
    if not isinstance(raw_stages, list):
        raise TrainingInputError("stages must be an array")

    stages: list[TrainingStage] = []
    for raw_stage in raw_stages:
        stage = _require_mapping(raw_stage, label="stage")
        mix = _require_mapping(stage.get("data_mix"), label="data_mix")
        stages.append(
            TrainingStage.create(
                kind=StageKind(_require_string(stage, "kind")),
                seed=stage.get("seed"),
                data_mix=DataMix(
                    code=float(mix.get("code", 0)),
                    agentic_tools=float(mix.get("agentic_tools", 0)),
                    russian_text=float(mix.get("russian_text", 0)),
                    english_text=float(mix.get("english_text", 0)),
                ),
                max_steps=stage.get("max_steps"),
                checkpoint_every=stage.get("checkpoint_every"),
            )
        )

    target_modules = lora.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or not all(isinstance(item, str) and item.strip() for item in target_modules)
    ):
        raise TrainingInputError("invalid LoRA target_modules")
    allowed_licenses = licenses.get("allowed")
    forbidden_licenses = licenses.get("forbidden")
    if (
        not isinstance(allowed_licenses, list)
        or not allowed_licenses
        or not all(isinstance(item, str) and item.strip() for item in allowed_licenses)
        or not isinstance(forbidden_licenses, list)
        or not all(isinstance(item, str) and item.strip() for item in forbidden_licenses)
    ):
        raise TrainingInputError("invalid license policy")

    config = MaintainerConfig(
        product_id=_require_string(document, "product_id"),
        document_sha256=_sha256_bytes(canonical),
        config_sha256_env=_require_string(document, "config_sha256_env"),
        source=EnvironmentReference(
            path_env=_require_string(source, "path_env"),
            sha256_env=_require_string(source, "sha256_env"),
        ),
        dataset=DatasetReference(
            manifest_env=_require_string(dataset, "manifest_env"),
            sha256_env=_require_string(dataset, "sha256_env"),
        ),
        output_dir=_require_string(document, "output_dir"),
        seed=int(document.get("seed", -1)),
        sequence_length=int(document.get("sequence_length", 0)),
        effective_batch_size=int(document.get("effective_batch_size", 0)),
        per_device_train_batch_size=int(document.get("per_device_train_batch_size", 0)),
        learning_rate=float(document.get("learning_rate", 0)),
        epochs=int(document.get("epochs", 0)),
        precision=_require_string(document, "precision"),
        license_policy=LicensePolicy(
            allowed_license_ids=tuple(item.strip() for item in allowed_licenses),
            forbidden_license_ids=tuple(item.strip() for item in forbidden_licenses),
        ),
        lora=LoraSettings(
            rank=int(lora.get("rank", 0)),
            alpha=int(lora.get("alpha", 0)),
            dropout=float(lora.get("dropout", -1)),
            target_modules=tuple(item.strip() for item in target_modules),
        ),
        stages=tuple(stages),
    )
    _validate_maintainer_config(config)
    return config


def _validate_maintainer_config(config: MaintainerConfig) -> None:
    if config.product_id != "metaflora-incubus-v1":
        raise TrainingInputError("unexpected product_id")
    if config.seed < 0 or config.per_device_train_batch_size <= 0:
        raise TrainingInputError("invalid deterministic batch settings")
    if config.effective_batch_size % config.per_device_train_batch_size:
        raise TrainingInputError("effective batch size must be divisible by per-device batch size")
    if config.lora.rank <= 0 or config.lora.alpha <= 0 or not 0 <= config.lora.dropout < 1:
        raise TrainingInputError("invalid LoRA settings")
    # Reuse the canonical contract validation and enforce exact stage order.
    provisional = TrainingConfig.create(
        seed=config.seed,
        dataset_sha256="0" * 64,
        source_artifact_sha256="0" * 64,
        sequence_length=config.sequence_length,
        effective_batch_size=config.effective_batch_size,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        precision=config.precision,
    )
    TrainingPlan.create(config=provisional, stages=config.stages)


def _jsonl_bytes(records: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json(record) + b"\n" for record in records)


def prepare_dataset(
    *,
    input_path: Path | str,
    output_dir: Path | str,
    expected_input_sha256: str,
    config_path: Path | str = "configs/training/incubus-v1.json",
    dry_run: bool = False,
) -> PreparedDataset:
    """Validate provenance and produce deterministic train-only JSONL surfaces."""

    source_path = Path(input_path)
    expected_hash = _require_sha256(expected_input_sha256, label="input")
    actual_hash = _sha256_file(source_path)
    if actual_hash != expected_hash:
        raise HashMismatchError("input sha256 mismatch")

    maintainer_config = load_maintainer_config(config_path)
    sft: list[dict[str, object]] = []
    preference: list[dict[str, object]] = []
    sft_validation: list[dict[str, object]] = []
    preference_validation: list[dict[str, object]] = []
    train_records: list[DatasetRecord] = []
    validation_records: list[DatasetRecord] = []
    provenance: list[dict[str, object]] = []
    catalog = DatasetCatalog.empty()
    seen_ids: set[str] = set()
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TrainingInputError("cannot read dataset input") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = _require_mapping(json.loads(line), label=f"line {line_number}")
            split = DatasetSplit(_require_string(raw, "split"))
            record = DatasetRecord.create(
                record_id=_require_string(raw, "record_id"),
                prompt=_require_string(raw, "prompt"),
                response=_require_string(raw, "response"),
                source_url=_require_string(raw, "source_url"),
                source_revision=_require_string(raw, "source_revision"),
                collected_at=_require_string(raw, "collected_at"),
                license_id=_require_string(raw, "license_id"),
                split=split,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingInputError(f"invalid dataset record at line {line_number}") from exc
        if split is DatasetSplit.HOLDOUT:
            raise TrainingInputError("holdout records are forbidden in maintainer training input")
        if not maintainer_config.license_policy.accepts(record):
            raise TrainingInputError(f"license is not allowed at line {line_number}")
        if record.record_id in seen_ids:
            raise TrainingInputError(f"duplicate record_id: {record.record_id}")
        seen_ids.add(record.record_id)
        try:
            catalog = catalog.ingest(record)
        except DuplicateExampleError as exc:
            raise TrainingInputError(f"duplicate normalized content at line {line_number}") from exc
        capability = raw.get("capability")
        if capability not in _CAPABILITIES:
            raise TrainingInputError(f"invalid capability at line {line_number}")
        provenance.append(
            {
                "capability": capability,
                "collected_at": record.collected_at,
                "content_sha256": record.content_sha256,
                "license_id": record.license_id,
                "record_id": record.record_id,
                "source_revision": record.source_revision,
                "source_url": record.source_url,
                "split": record.split.value,
            }
        )
        chosen = raw.get("chosen")
        rejected = raw.get("rejected")
        if not isinstance(chosen, str) or not chosen.strip():
            raise TrainingInputError(f"invalid chosen response at line {line_number}")
        if (
            not isinstance(rejected, str)
            or not rejected.strip()
            or chosen.strip() == rejected.strip()
        ):
            raise TrainingInputError(f"invalid rejected response at line {line_number}")
        sft_item = {
            "capability": capability,
            "content_sha256": record.content_sha256,
            "messages": [
                {"content": record.prompt, "role": "user"},
                {"content": record.response, "role": "assistant"},
            ],
            "record_id": record.record_id,
        }
        preference_item = {
            "capability": capability,
            "chosen": [{"content": chosen.strip(), "role": "assistant"}],
            "content_sha256": record.content_sha256,
            "prompt": [{"content": record.prompt, "role": "user"}],
            "record_id": record.record_id,
            "rejected": [{"content": rejected.strip(), "role": "assistant"}],
        }
        if split is DatasetSplit.VALIDATION:
            validation_records.append(record)
            sft_validation.append(sft_item)
            preference_validation.append(preference_item)
            continue
        train_records.append(record)
        sft.append(sft_item)
        preference.append(preference_item)

    if not sft or not preference or not sft_validation or not preference_validation:
        raise TrainingInputError("SFT, preference, and validation records are all required")
    try:
        DatasetPartition.validate(
            train=tuple(train_records),
            validation=tuple(validation_records),
            holdout=(),
            holdout_revision="no-holdout-training-input-v1",
        )
        scan_contamination(
            training_records=tuple(train_records),
            benchmark_records=tuple(validation_records),
        ).require_clean()
    except (ContaminationError, ValueError) as exc:
        raise TrainingInputError("train and validation leakage detected") from exc
    groups = {
        "sft.jsonl": sorted(sft, key=lambda item: str(item["record_id"])),
        "preference.jsonl": sorted(preference, key=lambda item: str(item["record_id"])),
        "sft_validation.jsonl": sorted(sft_validation, key=lambda item: str(item["record_id"])),
        "preference_validation.jsonl": sorted(
            preference_validation, key=lambda item: str(item["record_id"])
        ),
        _PROVENANCE_FILE: sorted(provenance, key=lambda item: str(item["record_id"])),
    }
    capability_counts = {
        name: {
            capability: sum(item["capability"] == capability for item in records)
            for capability in _CAPABILITIES
        }
        for name, records in groups.items()
        if name in _TRAINING_FILES
    }
    if any(not count for counts in capability_counts.values() for count in counts.values()):
        raise TrainingInputError("every training surface must cover every configured capability")
    payloads = {name: _jsonl_bytes(records) for name, records in groups.items()}
    file_hashes = {name: _sha256_bytes(payload) for name, payload in payloads.items()}
    dataset_sha256 = _sha256_bytes(_canonical_json(file_hashes))
    manifest = {
        "dataset_sha256": dataset_sha256,
        "files": file_hashes,
        "input_sha256": actual_hash,
        "capability_counts": capability_counts,
        "record_counts": {
            "preference": len(preference),
            "preference_validation": len(preference_validation),
            "sft": len(sft),
            "sft_validation": len(sft_validation),
        },
        "schema_version": 1,
    }
    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            _atomic_write(destination / name, payload)
        _atomic_write(manifest_path, _canonical_json(manifest) + b"\n")
    return PreparedDataset(
        manifest_path=manifest_path,
        dataset_sha256=dataset_sha256,
        input_sha256=actual_hash,
        record_counts=MappingProxyType(dict(manifest["record_counts"])),
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _environment_value(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value.strip():
        raise TrainingInputError(f"required environment variable is missing: {name}")
    return value.strip()


def _sha256_safe_artifact(path: Path) -> str:
    if path.is_symlink():
        raise TrainingInputError("artifact cannot be a symbolic link")
    if path.is_file():
        if path.suffix != ".safetensors":
            raise TrainingInputError("artifact must use safetensors")
        return _sha256_file(path)
    if not path.is_dir():
        raise TrainingInputError("artifact does not exist")
    entries_on_disk = tuple(path.rglob("*"))
    if any(item.is_symlink() for item in entries_on_disk):
        raise TrainingInputError("artifact directory cannot contain symbolic links")
    files = tuple(sorted(item for item in entries_on_disk if item.is_file()))
    if not files or not any(item.suffix == ".safetensors" for item in files):
        raise TrainingInputError("artifact directory has no safetensors weights")
    if any(item.suffix in {".bin", ".pt", ".pth"} for item in files):
        raise TrainingInputError("artifact directory contains unsafe weight files")
    entries = [
        {"path": item.relative_to(path).as_posix(), "sha256": _sha256_file(item)} for item in files
    ]
    return _sha256_bytes(_canonical_json(entries))


def calculate_source_artifact_sha256(path: Path | str) -> str:
    """Return the deterministic hash maintainers must pin for a safe local source."""

    return _sha256_safe_artifact(Path(path))


def _sha256_source_artifact(path: Path) -> str:
    if not path.is_dir():
        raise TrainingInputError("source artifact must be a loadable local directory")
    if not (path / "config.json").is_file() or not any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    ):
        raise TrainingInputError("source directory is missing local model or tokenizer config")
    return _sha256_safe_artifact(path)


def _load_dataset_manifest(path: Path, expected_dataset_sha256: str) -> dict[str, object]:
    document, _canonical = _read_json_object(path)
    if document.get("schema_version") != 1:
        raise TrainingInputError("unsupported dataset manifest schema")
    dataset_sha = _require_sha256(str(document.get("dataset_sha256", "")), label="dataset")
    if dataset_sha != expected_dataset_sha256:
        raise HashMismatchError("dataset sha256 mismatch")
    files = _require_mapping(document.get("files"), label="dataset files")
    if set(files) != set(_MANIFEST_FILES):
        raise TrainingInputError("dataset manifest must contain only training surfaces")
    actual_file_hashes: dict[str, str] = {}
    for name in _MANIFEST_FILES:
        expected_file_hash = _require_sha256(str(files[name]), label=name)
        candidate = path.parent / name
        if candidate.is_symlink() or not candidate.is_file():
            raise TrainingInputError(f"missing prepared dataset file: {name}")
        actual = _sha256_file(candidate)
        if actual != expected_file_hash:
            raise HashMismatchError(f"dataset file sha256 mismatch: {name}")
        actual_file_hashes[name] = actual
    capability_counts = _require_mapping(
        document.get("capability_counts"), label="capability counts"
    )
    if set(capability_counts) != set(_TRAINING_FILES):
        raise TrainingInputError("capability counts do not cover every training surface")
    for name in _TRAINING_FILES:
        counts = _require_mapping(capability_counts[name], label=f"capability counts for {name}")
        if set(counts) != set(_CAPABILITIES) or any(
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
            for count in counts.values()
        ):
            raise TrainingInputError("capability counts must be positive for every data mix")
    if _sha256_bytes(_canonical_json(actual_file_hashes)) != dataset_sha:
        raise HashMismatchError("dataset manifest aggregate sha256 mismatch")
    return document


def build_training_run(
    *,
    config_path: Path | str,
    environment: Mapping[str, str] | None = None,
    resume_metadata_path: Path | str | None = None,
    dry_run: bool,
) -> TrainingRun:
    """Resolve and verify a candidate run without importing the ML stack."""

    config = load_maintainer_config(config_path)
    values = os.environ if environment is None else environment
    expected_config_hash = _require_sha256(
        _environment_value(values, config.config_sha256_env), label="config"
    )
    if config.document_sha256 != expected_config_hash:
        raise HashMismatchError("config sha256 mismatch")

    source_path = Path(_environment_value(values, config.source.path_env)).expanduser()
    expected_source_hash = _require_sha256(
        _environment_value(values, config.source.sha256_env), label="source artifact"
    )
    actual_source_hash = _sha256_source_artifact(source_path)
    if actual_source_hash != expected_source_hash:
        raise HashMismatchError("source artifact sha256 mismatch")

    manifest_path = Path(_environment_value(values, config.dataset.manifest_env)).expanduser()
    expected_dataset_hash = _require_sha256(
        _environment_value(values, config.dataset.sha256_env), label="dataset"
    )
    _load_dataset_manifest(manifest_path, expected_dataset_hash)

    contract_config = TrainingConfig.create(
        seed=config.seed,
        dataset_sha256=expected_dataset_hash,
        source_artifact_sha256=actual_source_hash,
        sequence_length=config.sequence_length,
        effective_batch_size=config.effective_batch_size,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        precision=config.precision,
    )
    plan = TrainingPlan.create(config=contract_config, stages=config.stages)
    execution_config_sha256 = config.document_sha256
    execution_plan_sha256 = _sha256_bytes(
        _canonical_json(
            {
                "contract_plan_sha256": plan.plan_sha256,
                "execution_config_sha256": execution_config_sha256,
            }
        )
    )
    resume = None
    resume_path = None
    if resume_metadata_path is not None:
        metadata_path = Path(resume_metadata_path)
        raw_checkpoint, _canonical = _read_json_object(metadata_path)
        try:
            if raw_checkpoint.get("execution_config_sha256") != execution_config_sha256:
                raise HashMismatchError("resume execution config sha256 mismatch")
            if raw_checkpoint.get("execution_plan_sha256") != execution_plan_sha256:
                raise HashMismatchError("resume execution plan sha256 mismatch")
            checkpoint_document = {
                key: value
                for key, value in raw_checkpoint.items()
                if key not in {"execution_config_sha256", "execution_plan_sha256"}
            }
            checkpoint = CheckpointRef.create(
                **{
                    **checkpoint_document,
                    "stage": StageKind(_require_string(checkpoint_document, "stage")),
                }
            )
            resume = create_resume_plan(plan, checkpoint)
            resume_root = metadata_path.parent.resolve()
            resume_path = (resume_root / checkpoint.path).resolve()
            if resume_root not in resume_path.parents:
                raise TrainingInputError("resume checkpoint path escapes its metadata directory")
            if _sha256_safe_artifact(resume_path) != checkpoint.checkpoint_sha256:
                raise HashMismatchError("checkpoint sha256 mismatch")
        except TrainingInputError:
            raise
        except (CheckpointCompatibilityError, OSError, TypeError, ValueError) as exc:
            raise HashMismatchError(
                "resume metadata does not match the exact training plan"
            ) from exc

    output_dir = Path(config.output_dir)
    recipes = tuple(
        StageRecipe(
            kind=stage.kind,
            trainer="SFTTrainer" if stage.kind is StageKind.SFT else "DPOTrainer",
            seed=stage.seed,
            dataset_path=manifest_path.parent
            / ("sft.jsonl" if stage.kind is StageKind.SFT else "preference.jsonl"),
            validation_path=manifest_path.parent
            / (
                "sft_validation.jsonl"
                if stage.kind is StageKind.SFT
                else "preference_validation.jsonl"
            ),
            output_dir=output_dir / stage.kind.value,
            max_steps=stage.max_steps,
            checkpoint_every=stage.checkpoint_every,
            data_mix=stage.data_mix,
        )
        for stage in config.stages
    )
    return TrainingRun(
        product_id=config.product_id,
        dry_run=dry_run,
        plan=plan,
        source_path=source_path,
        dataset_manifest_path=manifest_path,
        output_dir=output_dir,
        execution_config_sha256=execution_config_sha256,
        execution_plan_sha256=execution_plan_sha256,
        model_load_kwargs=MappingProxyType(
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
            }
        ),
        stage_recipes=recipes,
        resume_checkpoint=resume,
        resume_path=resume_path,
        maintainer_config=config,
        candidate_state=CandidateState.ADAPTER_ONLY,
        release_ready=False,
        post_training=PostTrainingRecipe(
            input_state=CandidateState.ADAPTER_ONLY,
            steps=(
                "merge_adapter_safetensors",
                "export_gguf",
                "quantize_q5",
                "run_parity_and_release_gates",
            ),
        ),
    )


def pending_stage_recipes(run: TrainingRun) -> tuple[StageRecipe, ...]:
    """Return stages that remain, without replaying stages before a resume target."""

    if run.resume_checkpoint is None:
        return run.stage_recipes
    start = next(
        index
        for index, recipe in enumerate(run.stage_recipes)
        if recipe.kind is run.resume_checkpoint.stage
    )
    return run.stage_recipes[start:]


def train_candidate(
    *,
    config_path: Path | str,
    environment: Mapping[str, str] | None = None,
    resume_metadata_path: Path | str | None = None,
    dry_run: bool = False,
) -> TrainingRun:
    """Validate a run, then execute deterministic SFT and preference stages."""

    run = build_training_run(
        config_path=config_path,
        environment=environment,
        resume_metadata_path=resume_metadata_path,
        dry_run=dry_run,
    )
    if not dry_run:
        _execute_training(run)
    return run


def _execute_training(run: TrainingRun) -> None:
    try:
        import torch
        from accelerate import Accelerator
        from datasets import interleave_datasets, load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    except ImportError as exc:
        raise MissingTrainingDependencyError(
            "install the maintainer training extra before a real run"
        ) from exc

    config = run.maintainer_config
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(config.seed)
    set_seed(config.seed, deterministic=True)
    torch.use_deterministic_algorithms(True)
    accelerator = Accelerator(mixed_precision=config.precision)
    if accelerator.num_processes < 1:
        raise RuntimeError("Accelerate did not initialize a training process")

    model = AutoModelForCausalLM.from_pretrained(
        str(run.source_path),
        torch_dtype=torch.bfloat16 if config.precision == "bf16" else torch.float16,
        **dict(run.model_load_kwargs),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(run.source_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        task_type="CAUSAL_LM",
    )

    def load_mixed_dataset(path: Path, mix: DataMix, seed: int):
        raw_dataset = load_dataset("json", data_files=str(path), split="train")
        probabilities = [getattr(mix, capability) for capability in _CAPABILITIES]
        partitions = tuple(
            raw_dataset.filter(
                lambda example, expected=capability: example["capability"] == expected
            )
            for capability in _CAPABILITIES
        )
        if any(len(partition) == 0 for partition in partitions):
            raise TrainingInputError("prepared data cannot satisfy the configured data mix")
        return interleave_datasets(
            partitions,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    resume_path = str(run.resume_path) if run.resume_path is not None else None
    distributed_batch = config.per_device_train_batch_size * accelerator.num_processes
    if config.effective_batch_size % distributed_batch:
        raise TrainingInputError(
            "effective batch size is incompatible with the Accelerate process count"
        )
    has_existing_adapter = False
    for recipe in pending_stage_recipes(run):
        set_seed(recipe.seed, deterministic=True)
        dataset = load_mixed_dataset(recipe.dataset_path, recipe.data_mix, recipe.seed)
        validation = load_mixed_dataset(recipe.validation_path, recipe.data_mix, recipe.seed)
        common = {
            "output_dir": str(recipe.output_dir),
            "seed": recipe.seed,
            "data_seed": recipe.seed,
            "max_steps": recipe.max_steps,
            "save_steps": recipe.checkpoint_every,
            "learning_rate": config.learning_rate,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.effective_batch_size // distributed_batch,
            "save_safetensors": True,
            "full_determinism": True,
            "report_to": "none",
            "bf16": config.precision == "bf16",
            "fp16": config.precision == "fp16",
            "gradient_checkpointing": True,
        }
        if recipe.kind is StageKind.SFT:
            args = SFTConfig(max_length=config.sequence_length, **common)
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                eval_dataset=validation,
                processing_class=tokenizer,
                peft_config=peft_config,
            )
        else:
            args = DPOConfig(max_length=config.sequence_length, **common)
            dpo_kwargs = {
                "model": model,
                "args": args,
                "train_dataset": dataset,
                "eval_dataset": validation,
                "processing_class": tokenizer,
            }
            if not has_existing_adapter:
                dpo_kwargs["peft_config"] = peft_config
            trainer = DPOTrainer(
                **dpo_kwargs,
            )
        stage_resume = (
            resume_path
            if run.resume_checkpoint and run.resume_checkpoint.stage is recipe.kind
            else None
        )
        trainer.train(resume_from_checkpoint=stage_resume)
        trainer.save_model(str(recipe.output_dir / "final"))
        model = trainer.model
        has_existing_adapter = True
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        run.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            run.output_dir / "candidate-state.json",
            _canonical_json(
                {
                    "candidate_state": run.candidate_state.value,
                    "execution_plan_sha256": run.execution_plan_sha256,
                    "product_id": run.product_id,
                    "release_ready": run.release_ready,
                    "required_post_training": list(run.post_training.steps),
                }
            )
            + b"\n",
        )
