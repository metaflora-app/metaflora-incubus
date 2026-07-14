"""Pure, reproducible contracts for the Incubus v1 training and compact build."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from metaflora_incubus.benchmark_harness import (
    BenchmarkProvenance,
    HarnessReport,
)

GIB = 1024**3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION40 = re.compile(r"^[0-9a-f]{40}$")


class ProvenanceError(ValueError):
    pass


class DuplicateExampleError(ValueError):
    pass


class LeakageError(ValueError):
    pass


class ContaminationError(ValueError):
    pass


class CheckpointCompatibilityError(ValueError):
    pass


class ArtifactSizeError(ValueError):
    pass


class ReleaseEvidenceError(ValueError):
    pass


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


def _normalize(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(" ".join(line.split()) for line in lines if line.strip()).strip()


def _content_hash(prompt: str, response: str) -> str:
    payload = json.dumps(
        {
            "prompt": _normalize(prompt).casefold(),
            "response": _normalize(response).casefold(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    prompt: str
    response: str
    source_url: str
    source_revision: str
    collected_at: str
    license_id: str
    split: DatasetSplit
    content_sha256: str

    @classmethod
    def create(cls, **values: object) -> DatasetRecord:
        for field in (
            "record_id",
            "prompt",
            "response",
            "license_id",
        ):
            value = values.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ProvenanceError(f"invalid {field}")
        source_revision = values.get("source_revision")
        if not isinstance(source_revision, str) or _REVISION40.fullmatch(source_revision) is None:
            raise ProvenanceError("invalid source_revision")
        source_url = values.get("source_url")
        if not isinstance(source_url, str):
            raise ProvenanceError("invalid source_url")
        parsed = urlsplit(source_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ProvenanceError("invalid source_url")
        collected_at = values.get("collected_at")
        if not isinstance(collected_at, str):
            raise ProvenanceError("invalid collected_at")
        try:
            datetime.strptime(collected_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ProvenanceError("invalid collected_at") from exc
        split = values.get("split")
        if not isinstance(split, DatasetSplit):
            raise ProvenanceError("invalid split")
        prompt = str(values["prompt"])
        response = str(values["response"])
        return cls(
            record_id=str(values["record_id"]).strip(),
            prompt=_normalize(prompt),
            response=_normalize(response),
            source_url=source_url,
            source_revision=source_revision,
            collected_at=collected_at,
            license_id=str(values["license_id"]).strip(),
            split=split,
            content_sha256=_content_hash(prompt, response),
        )


@dataclass(frozen=True)
class DatasetCatalog:
    records: tuple[DatasetRecord, ...]

    @classmethod
    def empty(cls) -> DatasetCatalog:
        return cls(())

    def ingest(self, record: DatasetRecord) -> DatasetCatalog:
        if any(item.content_sha256 == record.content_sha256 for item in self.records):
            raise DuplicateExampleError(record.content_sha256)
        return DatasetCatalog((*self.records, record))


@dataclass(frozen=True)
class LicensePolicy:
    allowed_license_ids: tuple[str, ...]
    forbidden_license_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_license_ids", tuple(self.allowed_license_ids))
        object.__setattr__(self, "forbidden_license_ids", tuple(self.forbidden_license_ids))

    def accepts(self, record: DatasetRecord) -> bool:
        return (
            record.license_id in self.allowed_license_ids
            and record.license_id not in self.forbidden_license_ids
        )


@dataclass(frozen=True)
class PartitionSlice:
    records: tuple[DatasetRecord, ...]
    frozen: bool
    revision: str | None

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(item.record_id for item in self.records)


@dataclass(frozen=True)
class DatasetPartition:
    train: PartitionSlice
    validation: PartitionSlice
    holdout: PartitionSlice

    @classmethod
    def validate(
        cls,
        *,
        train: tuple[DatasetRecord, ...],
        validation: tuple[DatasetRecord, ...],
        holdout: tuple[DatasetRecord, ...],
        holdout_revision: str,
    ) -> DatasetPartition:
        groups = (tuple(train), tuple(validation), tuple(holdout))
        ids = [set(item.record_id for item in group) for group in groups]
        hashes = [set(item.content_sha256 for item in group) for group in groups]
        if any(ids[left] & ids[right] for left in range(3) for right in range(left + 1, 3)):
            raise LeakageError("record id leakage across dataset partitions")
        if any(hashes[left] & hashes[right] for left in range(3) for right in range(left + 1, 3)):
            raise LeakageError("content hash leakage across dataset partitions")
        if not holdout_revision.strip():
            raise LeakageError("holdout revision is required")
        return cls(
            PartitionSlice(groups[0], False, None),
            PartitionSlice(groups[1], False, None),
            PartitionSlice(groups[2], True, holdout_revision.strip()),
        )


def build_partitions(
    catalog: DatasetCatalog,
    *,
    seed: int,
    validation_fraction: float,
    holdout_fraction: float,
    holdout_revision: str,
) -> DatasetPartition:
    if seed < 0 or not 0 < validation_fraction < 1 or not 0 < holdout_fraction < 1:
        raise ValueError("invalid partition settings")
    if validation_fraction + holdout_fraction >= 1:
        raise ValueError("validation and holdout leave no training data")
    ordered = list(sorted(catalog.records, key=lambda item: item.content_sha256))
    random.Random(seed).shuffle(ordered)
    validation_count = round(len(ordered) * validation_fraction)
    holdout_count = round(len(ordered) * holdout_fraction)
    holdout = tuple(ordered[:holdout_count])
    validation = tuple(ordered[holdout_count : holdout_count + validation_count])
    train = tuple(ordered[holdout_count + validation_count :])
    return DatasetPartition.validate(
        train=train,
        validation=validation,
        holdout=holdout,
        holdout_revision=holdout_revision,
    )


@dataclass(frozen=True)
class TeacherCandidate:
    teacher_id: str
    response: str
    score: float

    @classmethod
    def create(cls, teacher_id: str, response: str, score: float) -> TeacherCandidate:
        if not teacher_id.strip():
            raise ValueError("teacher_id is required")
        if not response.strip() or not math.isfinite(score):
            raise ValueError("candidate response and finite score are required")
        return cls(teacher_id.strip(), response.strip(), score)


@dataclass(frozen=True)
class TeacherRankingInput:
    example: DatasetRecord
    candidates: tuple[TeacherCandidate, ...]

    @classmethod
    def create(
        cls,
        *,
        example: DatasetRecord,
        candidates: tuple[TeacherCandidate, ...],
    ) -> TeacherRankingInput:
        if example.split is DatasetSplit.HOLDOUT:
            raise LeakageError("holdout cannot be used for ranking")
        teacher_ids = [item.teacher_id for item in candidates]
        if len(candidates) < 2 or len(set(teacher_ids)) != len(teacher_ids):
            raise ValueError("teacher_id values must be distinct")
        return cls(example, tuple(sorted(candidates, key=lambda item: item.score, reverse=True)))


@dataclass(frozen=True)
class ContaminationMatch:
    training_record_id: str
    benchmark_record_id: str


@dataclass(frozen=True)
class ContaminationReport:
    clean: bool
    matches: tuple[ContaminationMatch, ...]

    def require_clean(self) -> None:
        if not self.clean:
            raise ContaminationError("benchmark contamination detected")


def scan_contamination(
    *,
    training_records: tuple[DatasetRecord, ...],
    benchmark_records: tuple[DatasetRecord, ...],
) -> ContaminationReport:
    matches = tuple(
        ContaminationMatch(training.record_id, benchmark.record_id)
        for training in training_records
        for benchmark in benchmark_records
        if _normalize(training.prompt).casefold() == _normalize(benchmark.prompt).casefold()
    )
    return ContaminationReport(not matches, matches)


class StageKind(str, Enum):
    SFT = "sft"
    PREFERENCE_DISTILLATION = "preference_distillation"


@dataclass(frozen=True)
class DataMix:
    code: float
    agentic_tools: float
    russian_text: float
    english_text: float

    def __post_init__(self) -> None:
        values = (self.code, self.agentic_tools, self.russian_text, self.english_text)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("all data mix values must be positive")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("data mix values must sum to one")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    dataset_sha256: str
    source_artifact_sha256: str
    sequence_length: int
    effective_batch_size: int
    learning_rate: float
    epochs: int
    precision: str
    config_sha256: str

    @classmethod
    def create(cls, **values: object) -> TrainingConfig:
        if not isinstance(values.get("seed"), int) or int(values["seed"]) < 0:
            raise ValueError("invalid seed")
        for field in ("dataset_sha256", "source_artifact_sha256"):
            if not isinstance(values.get(field), str) or not _SHA256.fullmatch(str(values[field])):
                raise ValueError(f"invalid {field}")
        for field in ("sequence_length", "effective_batch_size", "epochs"):
            value = values.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"invalid {field}")
        learning_rate = values.get("learning_rate")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("invalid learning_rate")
        if values.get("precision") not in {"bf16", "fp16"}:
            raise ValueError("invalid precision")
        canonical = {key: values[key] for key in sorted(values)}
        config_hash = _canonical_hash(canonical)
        return cls(**canonical, config_sha256=config_hash)


@dataclass(frozen=True)
class TrainingStage:
    kind: StageKind
    seed: int
    data_mix: DataMix
    max_steps: int
    checkpoint_every: int

    @classmethod
    def create(cls, **values: object) -> TrainingStage:
        if not isinstance(values.get("kind"), StageKind):
            raise ValueError("invalid stage kind")
        for field in ("seed", "max_steps", "checkpoint_every"):
            value = values.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"invalid {field}")
        if values["checkpoint_every"] > values["max_steps"]:
            raise ValueError("checkpoint interval exceeds stage")
        return cls(**values)


@dataclass(frozen=True)
class TrainingPlan:
    config: TrainingConfig
    stages: tuple[TrainingStage, ...]
    plan_sha256: str

    @classmethod
    def create(cls, *, config: TrainingConfig, stages: tuple[TrainingStage, ...]) -> TrainingPlan:
        kinds = tuple(item.kind for item in stages)
        if StageKind.SFT not in kinds:
            raise ValueError("SFT stage is required")
        required = (StageKind.SFT, StageKind.PREFERENCE_DISTILLATION)
        if kinds != required:
            raise ValueError("training stage order must be SFT then preference distillation")
        payload = {
            "config_sha256": config.config_sha256,
            "stages": [
                {
                    "kind": item.kind.value,
                    "seed": item.seed,
                    "data_mix": item.data_mix.__dict__,
                    "max_steps": item.max_steps,
                    "checkpoint_every": item.checkpoint_every,
                }
                for item in stages
            ],
        }
        return cls(config, tuple(stages), _canonical_hash(payload))


@dataclass(frozen=True)
class CheckpointRef:
    path: str
    stage: StageKind
    completed_step: int
    checkpoint_sha256: str
    plan_sha256: str
    config_sha256: str
    dataset_sha256: str
    source_artifact_sha256: str

    @classmethod
    def create(cls, **values: object) -> CheckpointRef:
        path = values.get("path")
        if (
            not isinstance(path, str)
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
        ):
            raise ValueError("invalid checkpoint path")
        if not isinstance(values.get("stage"), StageKind):
            raise ValueError("invalid checkpoint stage")
        step = values.get("completed_step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("invalid completed_step")
        for field in (
            "checkpoint_sha256",
            "plan_sha256",
            "config_sha256",
            "dataset_sha256",
            "source_artifact_sha256",
        ):
            if not isinstance(values.get(field), str) or not _SHA256.fullmatch(str(values[field])):
                raise ValueError(f"invalid {field}")
        return cls(**values)


@dataclass(frozen=True)
class ResumePlan:
    checkpoint: CheckpointRef
    stage: StageKind
    next_step: int


def create_resume_plan(plan: TrainingPlan, checkpoint: CheckpointRef) -> ResumePlan:
    expected = {
        "plan_sha256": plan.plan_sha256,
        "config_sha256": plan.config.config_sha256,
        "dataset_sha256": plan.config.dataset_sha256,
        "source_artifact_sha256": plan.config.source_artifact_sha256,
    }
    for field, value in expected.items():
        if getattr(checkpoint, field) != value:
            raise CheckpointCompatibilityError(field)
    stage = next((item for item in plan.stages if item.kind is checkpoint.stage), None)
    if stage is None or checkpoint.completed_step > stage.max_steps:
        raise CheckpointCompatibilityError("completed_step")
    return ResumePlan(checkpoint, checkpoint.stage, checkpoint.completed_step + 1)


class QuantizationFormat(str, Enum):
    BF16 = "bf16"
    Q8 = "q8"
    Q6 = "q6"
    Q5 = "q5"
    Q4 = "q4"


@dataclass(frozen=True)
class ParityJob:
    baseline_format: QuantizationFormat
    candidate_format: QuantizationFormat
    required_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_metrics", tuple(self.required_metrics))


@dataclass(frozen=True)
class QuantizationMatrix:
    formats: tuple[QuantizationFormat, ...]
    deployable_format: QuantizationFormat
    parity_jobs: tuple[ParityJob, ...]

    @classmethod
    def default(cls) -> QuantizationMatrix:
        formats = (
            QuantizationFormat.BF16,
            QuantizationFormat.Q8,
            QuantizationFormat.Q6,
            QuantizationFormat.Q5,
            QuantizationFormat.Q4,
        )
        metrics = (
            "coding",
            "agentic_search",
            "text_quality",
            "russian",
            "english",
            "overrefusal_rate",
        )
        return cls.create(
            formats=formats,
            deployable_format=QuantizationFormat.Q5,
            parity_jobs=tuple(
                ParityJob(QuantizationFormat.BF16, candidate, metrics) for candidate in formats[1:]
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        formats: tuple[QuantizationFormat, ...],
        deployable_format: QuantizationFormat,
        parity_jobs: tuple[ParityJob, ...],
    ) -> QuantizationMatrix:
        if deployable_format is not QuantizationFormat.Q5:
            raise ValueError("Q5 must be the deployable format")
        if QuantizationFormat.Q5 not in formats or not any(
            item.candidate_format is QuantizationFormat.Q5
            and item.baseline_format is QuantizationFormat.BF16
            for item in parity_jobs
        ):
            raise ValueError("Q5 parity against BF16 is required")
        return cls(tuple(formats), deployable_format, tuple(parity_jobs))


@dataclass(frozen=True)
class BuildArtifact:
    artifact_id: str
    format: QuantizationFormat
    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def create(cls, **values: object) -> BuildArtifact:
        if (
            values.get("format") is QuantizationFormat.Q5
            and not 3 * GIB <= int(values["size_bytes"]) <= 5 * GIB
        ):
            raise ArtifactSizeError("Q5 artifact must be between 3 and 5 GiB")
        if not isinstance(values.get("sha256"), str) or not _SHA256.fullmatch(
            str(values["sha256"])
        ):
            raise ValueError("invalid artifact SHA-256")
        return cls(**values)


class ComputeProfile(str, Enum):
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"


@dataclass(frozen=True)
class ResourceEstimateInput:
    parameter_count: int
    trainable_parameter_count: int
    sequence_length: int
    effective_batch_size: int
    dataset_tokens: int
    checkpoint_count: int
    target_format: QuantizationFormat


@dataclass(frozen=True)
class ResourceProfile:
    gpu_vram_bytes: int
    ram_bytes: int
    disk_bytes: int


@dataclass(frozen=True)
class ResourceEstimate:
    training: ResourceProfile
    build: ResourceProfile
    runtime: ResourceProfile
    recommended_profile: ComputeProfile


def estimate_resources(request: ResourceEstimateInput) -> ResourceEstimate:
    for field in (
        "parameter_count",
        "trainable_parameter_count",
        "sequence_length",
        "effective_batch_size",
        "dataset_tokens",
        "checkpoint_count",
    ):
        if getattr(request, field) <= 0:
            raise ValueError(field)
    parameter_bytes = request.parameter_count * 2
    optimizer_bytes = request.trainable_parameter_count * 12
    activation_bytes = request.sequence_length * request.effective_batch_size * 4096
    training_vram = parameter_bytes + optimizer_bytes + activation_bytes
    build_disk = max(6 * GIB, parameter_bytes * 2)
    training_disk = build_disk + request.checkpoint_count * (parameter_bytes + optimizer_bytes)
    runtime_disk = 5 * GIB if request.target_format is QuantizationFormat.Q5 else parameter_bytes
    return ResourceEstimate(
        training=ResourceProfile(training_vram, training_vram * 2, training_disk),
        build=ResourceProfile(parameter_bytes, parameter_bytes * 2, build_disk),
        runtime=ResourceProfile(0, max(12 * GIB, runtime_disk * 2), runtime_disk),
        recommended_profile=(
            ComputeProfile.SINGLE_GPU if training_vram <= 80 * GIB else ComputeProfile.MULTI_GPU
        ),
    )


@dataclass(frozen=True)
class BuildReleaseEvidence:
    artifact_sha256: str
    harness_report: HarnessReport
    provenance: BenchmarkProvenance
    raw_outputs_path: str

    @classmethod
    def create(cls, **values: object) -> BuildReleaseEvidence:
        report = values.get("harness_report")
        provenance = values.get("provenance")
        artifact_sha = values.get("artifact_sha256")
        if not isinstance(report, HarnessReport):
            raise ReleaseEvidenceError("harness_report must be a HarnessReport")
        if not report.approved:
            raise ReleaseEvidenceError("harness report is not approved")
        if not isinstance(provenance, BenchmarkProvenance):
            raise ReleaseEvidenceError("provenance is required")
        if artifact_sha != provenance.artifact_sha256:
            raise ReleaseEvidenceError("artifact_sha256 mismatch")
        raw_path = values.get("raw_outputs_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ReleaseEvidenceError("raw outputs path is required")
        return cls(**values)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()
