"""Immutable registry for externally reported, reproducible model weaknesses."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from urllib.parse import urlsplit


class ProvenanceError(ValueError):
    """Raised when an issue does not have safe, complete provenance."""


class DuplicateReproducerError(ValueError):
    """Raised when a semantically identical reproducer is already registered."""


class SourceType(str, Enum):
    OFFICIAL_ISSUE = "official_issue"
    HUGGING_FACE_DISCUSSION = "hugging_face_discussion"
    REDDIT = "reddit"
    FORUM = "forum"
    REPORT = "report"


class IssueCategory(str, Enum):
    UNJUSTIFIED_REFUSAL = "unjustified_refusal"
    CODE_CORRECTNESS = "code_correctness"
    TOOL_CALLING = "tool_calling"
    AGENTIC_COMPLETION = "agentic_completion"
    LONG_CONTEXT = "long_context"
    QUANTIZATION_REGRESSION = "quantization_regression"
    HALLUCINATION = "hallucination"
    REPETITION = "repetition"
    MULTILINGUAL_QUALITY = "multilingual_quality"
    RUNTIME_PERFORMANCE = "runtime_performance"


class IssueStatus(str, Enum):
    HYPOTHESIS = "hypothesis"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"


class ExportPurpose(str, Enum):
    TRAINING = "training"
    REGRESSION = "regression"


def _require_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ProvenanceError(f"{label} is required")
    return normalized


def _normalize_reproducer(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    normalized_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(line for line in normalized_lines if line).strip()


def create_reproducer_hash(reproducer: str) -> str:
    normalized = _normalize_reproducer(reproducer)
    if not normalized:
        raise ProvenanceError("reproducer is required")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvaluatorEvidence:
    evaluator_id: str
    evaluator_revision: str
    reproduced: bool
    rationale: str


@dataclass(frozen=True)
class Confirmation:
    baseline_output: str
    evaluator_evidence: EvaluatorEvidence


@dataclass(frozen=True)
class WeaknessHypothesis:
    source_type: SourceType
    source_url: str
    source_date: date
    source_model_revision: str
    category: IssueCategory
    reproducer: str
    reproducer_hash: str
    status: IssueStatus = IssueStatus.HYPOTHESIS
    confirmation: Confirmation | None = None
    resolution_revision: str | None = None

    @classmethod
    def create(
        cls,
        *,
        source_type: SourceType,
        source_url: str,
        source_date: date | None,
        source_model_revision: str,
        category: IssueCategory,
        reproducer: str,
    ) -> WeaknessHypothesis:
        url = _require_text(source_url, "source URL")
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise ProvenanceError("source URL must use HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise ProvenanceError("source URL must not contain credentials")
        if source_date is None:
            raise ProvenanceError("source date is required")
        revision = _require_text(source_model_revision, "model revision")
        normalized_reproducer = _normalize_reproducer(reproducer)
        if not normalized_reproducer:
            raise ProvenanceError("reproducer is required")
        return cls(
            source_type=source_type,
            source_url=url,
            source_date=source_date,
            source_model_revision=revision,
            category=category,
            reproducer=normalized_reproducer,
            reproducer_hash=create_reproducer_hash(normalized_reproducer),
        )


@dataclass(frozen=True)
class WeaknessRegistry:
    records: tuple[WeaknessHypothesis, ...]

    @classmethod
    def empty(cls) -> WeaknessRegistry:
        return cls(records=())

    def ingest(self, issue: WeaknessHypothesis) -> WeaknessRegistry:
        if any(record.reproducer_hash == issue.reproducer_hash for record in self.records):
            raise DuplicateReproducerError(f"duplicate reproducer hash: {issue.reproducer_hash}")
        return WeaknessRegistry(records=(*self.records, issue))

    def confirm(
        self,
        reproducer_hash: str,
        *,
        baseline_output: str,
        evaluator_evidence: EvaluatorEvidence,
    ) -> WeaknessRegistry:
        output = baseline_output.strip()
        if not output:
            raise ValueError("baseline output is required")
        if not evaluator_evidence.reproduced:
            raise ValueError("evaluator evidence must mark the issue as reproduced")
        return self._replace_record(
            reproducer_hash,
            lambda record: replace(
                record,
                status=IssueStatus.CONFIRMED,
                confirmation=Confirmation(output, evaluator_evidence),
            ),
        )

    def resolve(self, reproducer_hash: str, *, resolution_revision: str) -> WeaknessRegistry:
        revision = resolution_revision.strip()
        if not revision:
            raise ValueError("resolution revision is required")

        def resolve_record(record: WeaknessHypothesis) -> WeaknessHypothesis:
            if record.status is not IssueStatus.CONFIRMED:
                raise ValueError("only a confirmed issue can be resolved")
            return replace(
                record,
                status=IssueStatus.RESOLVED,
                resolution_revision=revision,
            )

        return self._replace_record(reproducer_hash, resolve_record)

    def _replace_record(self, reproducer_hash: str, transform) -> WeaknessRegistry:
        found = False
        updated: list[WeaknessHypothesis] = []
        for record in self.records:
            if record.reproducer_hash == reproducer_hash:
                updated.append(transform(record))
                found = True
            else:
                updated.append(record)
        if not found:
            raise KeyError(f"unknown reproducer hash: {reproducer_hash}")
        return WeaknessRegistry(records=tuple(updated))

    def export_jsonl(self, purpose: ExportPurpose) -> str:
        eligible = [
            record
            for record in self.records
            if record.status is IssueStatus.CONFIRMED
            or (purpose is ExportPurpose.REGRESSION and record.status is IssueStatus.RESOLVED)
        ]
        rows = [
            json.dumps(_as_export(record), ensure_ascii=False, sort_keys=True)
            for record in sorted(eligible, key=lambda item: item.reproducer_hash)
        ]
        return "" if not rows else "\n".join(rows) + "\n"


def _as_export(record: WeaknessHypothesis) -> dict[str, object]:
    confirmation = record.confirmation
    result: dict[str, object] = {
        "category": record.category.value,
        "reproducer": record.reproducer,
        "reproducer_hash": record.reproducer_hash,
        "source_date": record.source_date.isoformat(),
        "source_model_revision": record.source_model_revision,
        "source_type": record.source_type.value,
        "source_url": record.source_url,
        "status": record.status.value,
    }
    if confirmation is not None:
        result["baseline_output"] = confirmation.baseline_output
        result["evaluator_evidence"] = {
            "evaluator_id": confirmation.evaluator_evidence.evaluator_id,
            "evaluator_revision": confirmation.evaluator_evidence.evaluator_revision,
            "rationale": confirmation.evaluator_evidence.rationale,
            "reproduced": confirmation.evaluator_evidence.reproduced,
        }
    if record.resolution_revision is not None:
        result["resolution_revision"] = record.resolution_revision
    return result
