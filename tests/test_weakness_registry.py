import json
from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from metaflora_incubus.weakness_registry import (
    DuplicateReproducerError,
    EvaluatorEvidence,
    ExportPurpose,
    IssueCategory,
    IssueStatus,
    ProvenanceError,
    SourceType,
    WeaknessHypothesis,
    WeaknessRegistry,
    create_reproducer_hash,
)


def hypothesis(
    reproducer: str = "Explain why this allowed request was refused.",
    *,
    source_type: SourceType = SourceType.REDDIT,
    source_url: str = "https://www.reddit.com/r/LocalLLaMA/comments/example/report/",
    source_date: date | None = date(2026, 7, 14),
    source_model_revision: str = "revision-abc123",
    category: IssueCategory = IssueCategory.UNJUSTIFIED_REFUSAL,
) -> WeaknessHypothesis:
    return WeaknessHypothesis.create(
        source_type=source_type,
        source_url=source_url,
        source_date=source_date,
        source_model_revision=source_model_revision,
        category=category,
        reproducer=reproducer,
    )


def confirming_evidence() -> EvaluatorEvidence:
    return EvaluatorEvidence(
        evaluator_id="incubus-evaluator-v1",
        evaluator_revision="eval-42",
        reproduced=True,
        rationale="The baseline repeats the reported failure under the pinned revision.",
    )


@pytest.mark.parametrize(
    ("source_type", "source_url"),
    (
        (SourceType.OFFICIAL_ISSUE, "https://github.com/org/model/issues/42"),
        (SourceType.HUGGING_FACE_DISCUSSION, "https://huggingface.co/org/model/discussions/7"),
        (SourceType.REDDIT, "https://www.reddit.com/r/LocalLLaMA/comments/example/report/"),
        (SourceType.FORUM, "https://community.example.org/t/model-failure/17"),
        (SourceType.REPORT, "https://reports.example.org/incidents/model-2026-07"),
    ),
)
def test_ingests_hypotheses_from_each_allowed_https_source_type(
    source_type: SourceType,
    source_url: str,
) -> None:
    issue = hypothesis(source_type=source_type, source_url=source_url)

    registry = WeaknessRegistry.empty().ingest(issue)

    stored = registry.records[0]
    assert stored.source_type is source_type
    assert stored.source_url == source_url
    assert stored.source_date == date(2026, 7, 14)
    assert stored.source_model_revision == "revision-abc123"
    assert stored.category is IssueCategory.UNJUSTIFIED_REFUSAL
    assert stored.reproducer_hash == create_reproducer_hash(
        "Explain why this allowed request was refused."
    )
    assert stored.status is IssueStatus.HYPOTHESIS


@pytest.mark.parametrize(
    ("override", "expected_message"),
    (
        ({"source_url": "http://example.org/report/1"}, "HTTPS"),
        ({"source_url": "https://user:secret@example.org/report/1"}, "credentials"),
        ({"source_url": ""}, "source URL"),
        ({"source_date": None}, "source date"),
        ({"source_model_revision": ""}, "model revision"),
    ),
)
def test_rejects_missing_or_unsafe_provenance(
    override: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ProvenanceError, match=expected_message):
        hypothesis(**override)


def test_records_and_registry_are_immutable() -> None:
    issue = hypothesis()
    registry = WeaknessRegistry.empty().ingest(issue)

    with pytest.raises(FrozenInstanceError):
        issue.status = IssueStatus.CONFIRMED  # type: ignore[misc]
    with pytest.raises(AttributeError):
        registry.records.append(issue)  # type: ignore[attr-defined]


def test_ingest_returns_a_new_registry_without_mutating_the_previous_one() -> None:
    empty = WeaknessRegistry.empty()

    populated = empty.ingest(hypothesis())

    assert empty.records == ()
    assert len(populated.records) == 1


def test_normalized_reproducer_hash_deduplicates_formatting_variants() -> None:
    first = hypothesis("  line one\r\nline   two  ")
    formatting_variant = hypothesis(
        "line one\nline two",
        source_url="https://huggingface.co/org/model/discussions/99",
        source_type=SourceType.HUGGING_FACE_DISCUSSION,
    )
    registry = WeaknessRegistry.empty().ingest(first)

    with pytest.raises(DuplicateReproducerError, match=first.reproducer_hash):
        registry.ingest(formatting_variant)


def test_confirmation_requires_a_nonempty_baseline_output() -> None:
    registry = WeaknessRegistry.empty().ingest(hypothesis())
    issue_hash = registry.records[0].reproducer_hash

    with pytest.raises(ValueError, match="baseline output"):
        registry.confirm(
            issue_hash,
            baseline_output="   ",
            evaluator_evidence=confirming_evidence(),
        )


def test_confirmation_requires_positive_evaluator_evidence() -> None:
    registry = WeaknessRegistry.empty().ingest(hypothesis())
    issue_hash = registry.records[0].reproducer_hash
    negative_evidence = EvaluatorEvidence(
        evaluator_id="incubus-evaluator-v1",
        evaluator_revision="eval-42",
        reproduced=False,
        rationale="The reported behavior could not be reproduced.",
    )

    with pytest.raises(ValueError, match="reproduced"):
        registry.confirm(
            issue_hash,
            baseline_output="The baseline produced an acceptable response.",
            evaluator_evidence=negative_evidence,
        )


def test_confirmation_preserves_baseline_and_evaluator_evidence() -> None:
    registry = WeaknessRegistry.empty().ingest(hypothesis())
    issue_hash = registry.records[0].reproducer_hash

    confirmed = registry.confirm(
        issue_hash,
        baseline_output="I cannot help with that benign request.",
        evaluator_evidence=confirming_evidence(),
    )

    record = confirmed.records[0]
    assert record.status is IssueStatus.CONFIRMED
    assert record.confirmation is not None
    assert record.confirmation.baseline_output == "I cannot help with that benign request."
    assert record.confirmation.evaluator_evidence.reproduced is True
    assert registry.records[0].status is IssueStatus.HYPOTHESIS


def test_unconfirmed_hypotheses_are_excluded_from_all_exports() -> None:
    registry = WeaknessRegistry.empty().ingest(hypothesis())

    assert registry.export_jsonl(ExportPurpose.TRAINING) == ""
    assert registry.export_jsonl(ExportPurpose.REGRESSION) == ""


def test_confirmed_reproducible_issue_enters_training_and_regression_exports() -> None:
    registry = WeaknessRegistry.empty().ingest(hypothesis())
    issue_hash = registry.records[0].reproducer_hash
    confirmed = registry.confirm(
        issue_hash,
        baseline_output="Refusal reproduced.",
        evaluator_evidence=confirming_evidence(),
    )

    training = json.loads(confirmed.export_jsonl(ExportPurpose.TRAINING))
    regression = json.loads(confirmed.export_jsonl(ExportPurpose.REGRESSION))

    assert training["reproducer_hash"] == issue_hash
    assert regression["reproducer_hash"] == issue_hash
    assert training["status"] == "confirmed"


def test_resolved_issue_remains_a_permanent_regression_case() -> None:
    registry = WeaknessRegistry.empty().ingest(hypothesis())
    issue_hash = registry.records[0].reproducer_hash
    confirmed = registry.confirm(
        issue_hash,
        baseline_output="Refusal reproduced.",
        evaluator_evidence=confirming_evidence(),
    )

    resolved = confirmed.resolve(issue_hash, resolution_revision="incubus-v1-build-17")

    assert resolved.records[0].status is IssueStatus.RESOLVED
    assert resolved.records[0].confirmation is not None
    assert resolved.export_jsonl(ExportPurpose.TRAINING) == ""
    exported = json.loads(resolved.export_jsonl(ExportPurpose.REGRESSION))
    assert exported["reproducer_hash"] == issue_hash
    assert exported["status"] == "resolved"
    assert exported["resolution_revision"] == "incubus-v1-build-17"


def test_cannot_resolve_an_issue_that_was_never_confirmed() -> None:
    registry = WeaknessRegistry.empty().ingest(hypothesis())
    issue_hash = registry.records[0].reproducer_hash

    with pytest.raises(ValueError, match="confirmed"):
        registry.resolve(issue_hash, resolution_revision="incubus-v1-build-17")


def test_jsonl_export_is_deterministic_across_ingest_order() -> None:
    first = hypothesis(
        "First independently reproducible weakness",
        source_url="https://github.com/org/model/issues/1",
        source_type=SourceType.OFFICIAL_ISSUE,
    )
    second = hypothesis(
        "Second independently reproducible weakness",
        source_url="https://github.com/org/model/issues/2",
        source_type=SourceType.OFFICIAL_ISSUE,
        category=IssueCategory.CODE_CORRECTNESS,
    )

    def confirmed_registry(issues: tuple[WeaknessHypothesis, ...]) -> WeaknessRegistry:
        registry = WeaknessRegistry.empty()
        for issue in issues:
            registry = registry.ingest(issue)
            registry = registry.confirm(
                issue.reproducer_hash,
                baseline_output=f"Baseline output for {issue.reproducer_hash}",
                evaluator_evidence=confirming_evidence(),
            )
        return registry

    forward = confirmed_registry((first, second))
    reverse = confirmed_registry((second, first))

    forward_jsonl = forward.export_jsonl(ExportPurpose.REGRESSION)
    reverse_jsonl = reverse.export_jsonl(ExportPurpose.REGRESSION)
    assert forward_jsonl == reverse_jsonl
    assert forward_jsonl.endswith("\n")
    hashes = [json.loads(line)["reproducer_hash"] for line in forward_jsonl.splitlines()]
    assert hashes == sorted(hashes)
