from dataclasses import FrozenInstanceError

import pytest

from metaflora_incubus.training_contract import (
    ContaminationError,
    DatasetCatalog,
    DatasetPartition,
    DatasetRecord,
    DatasetSplit,
    DuplicateExampleError,
    LeakageError,
    LicensePolicy,
    ProvenanceError,
    TeacherCandidate,
    TeacherRankingInput,
    build_partitions,
    scan_contamination,
)


def record(
    record_id: str,
    prompt: str,
    response: str,
    *,
    split: DatasetSplit = DatasetSplit.TRAIN,
    license_id: str = "Apache-2.0",
) -> DatasetRecord:
    return DatasetRecord.create(
        record_id=record_id,
        prompt=prompt,
        response=response,
        source_url=f"https://datasets.example/{record_id}",
        source_revision="0123456789abcdef0123456789abcdef01234567",
        collected_at="2026-07-14T12:00:00Z",
        license_id=license_id,
        split=split,
    )


def test_dataset_record_is_immutable_and_contains_auditable_provenance() -> None:
    item = record("sample-001", "Implement a parser", "Here is the implementation")

    assert item.source_url == "https://datasets.example/sample-001"
    assert item.source_revision == "0123456789abcdef0123456789abcdef01234567"
    assert item.collected_at == "2026-07-14T12:00:00Z"
    assert item.license_id == "Apache-2.0"
    assert len(item.content_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        item.response = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("override", "field"),
    (
        ({"source_url": "http://datasets.example/item"}, "source_url"),
        ({"source_url": "https://token@datasets.example/item"}, "source_url"),
        ({"source_revision": ""}, "source_revision"),
        ({"collected_at": "yesterday"}, "collected_at"),
        ({"license_id": ""}, "license_id"),
    ),
)
def test_dataset_record_rejects_incomplete_or_unsafe_provenance(
    override: dict[str, str], field: str
) -> None:
    values = {
        "record_id": "sample-001",
        "prompt": "Implement a parser",
        "response": "Here is the implementation",
        "source_url": "https://datasets.example/item",
        "source_revision": "0123456789abcdef0123456789abcdef01234567",
        "collected_at": "2026-07-14T12:00:00Z",
        "license_id": "Apache-2.0",
        "split": DatasetSplit.TRAIN,
    }

    with pytest.raises(ProvenanceError, match=field):
        DatasetRecord.create(**{**values, **override})


@pytest.mark.parametrize("revision", ("main", "A" * 40, "a" * 39, "g" * 40))
def test_dataset_record_requires_pinned_lowercase_source_revision(revision: str) -> None:
    with pytest.raises(ProvenanceError, match="source_revision"):
        DatasetRecord.create(
            record_id="sample-001",
            prompt="Implement a parser",
            response="Here is the implementation",
            source_url="https://datasets.example/item",
            source_revision=revision,
            collected_at="2026-07-14T12:00:00Z",
            license_id="Apache-2.0",
            split=DatasetSplit.TRAIN,
        )


def test_catalog_deduplicates_normalized_prompt_response_content() -> None:
    first = record("sample-001", "  line one\r\nline   two ", " answer   one ")
    duplicate = record("sample-002", "line one\nline two", "answer one")
    catalog = DatasetCatalog.empty().ingest(first)

    with pytest.raises(DuplicateExampleError, match=first.content_sha256):
        catalog.ingest(duplicate)

    assert catalog.records == (first,)


def test_license_policy_fails_closed_for_unknown_or_forbidden_terms() -> None:
    policy = LicensePolicy(
        allowed_license_ids=("Apache-2.0", "MIT", "CC-BY-4.0"),
        forbidden_license_ids=("unknown", "all-rights-reserved"),
    )

    assert policy.accepts(record("allowed", "p", "r")) is True
    assert policy.accepts(record("unknown", "p2", "r2", license_id="unknown")) is False
    assert policy.accepts(record("unlisted", "p3", "r3", license_id="custom")) is False


def test_train_validation_and_frozen_holdout_are_disjoint() -> None:
    catalog = DatasetCatalog.empty()
    for index in range(30):
        catalog = catalog.ingest(record(f"sample-{index:03}", f"prompt {index}", f"reply {index}"))

    partitions = build_partitions(
        catalog,
        seed=1701,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        holdout_revision="holdout-v1",
    )

    assert partitions.train.ids.isdisjoint(partitions.validation.ids)
    assert partitions.train.ids.isdisjoint(partitions.holdout.ids)
    assert partitions.validation.ids.isdisjoint(partitions.holdout.ids)
    assert partitions.holdout.frozen is True
    assert partitions.holdout.revision == "holdout-v1"
    assert partitions == build_partitions(
        catalog,
        seed=1701,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        holdout_revision="holdout-v1",
    )


def test_partition_constructor_rejects_record_and_content_hash_leakage() -> None:
    train = record("train-001", "same prompt", "same response")
    leaked = record(
        "holdout-001",
        " same   prompt ",
        "same response",
        split=DatasetSplit.HOLDOUT,
    )

    with pytest.raises(LeakageError):
        DatasetPartition.validate(
            train=(train,),
            validation=(),
            holdout=(leaked,),
            holdout_revision="holdout-v1",
        )


def test_frozen_holdout_cannot_be_used_as_training_or_ranking_input() -> None:
    holdout = record(
        "holdout-001",
        "private evaluation prompt",
        "private expected answer",
        split=DatasetSplit.HOLDOUT,
    )

    with pytest.raises(LeakageError, match="holdout"):
        TeacherRankingInput.create(
            example=holdout,
            candidates=(
                TeacherCandidate.create("teacher-a", "candidate a", 0.7),
                TeacherCandidate.create("teacher-b", "candidate b", 0.8),
            ),
        )


def test_multi_teacher_ranking_is_generic_immutable_and_requires_distinct_candidates() -> None:
    ranking = TeacherRankingInput.create(
        example=record("rank-001", "Solve the task", "reference"),
        candidates=(
            TeacherCandidate.create("teacher-a", "candidate a", 0.65),
            TeacherCandidate.create("teacher-b", "candidate b", 0.88),
            TeacherCandidate.create("teacher-c", "candidate c", 0.73),
        ),
    )

    assert ranking.candidates[0].teacher_id == "teacher-b"
    assert tuple(item.score for item in ranking.candidates) == (0.88, 0.73, 0.65)
    with pytest.raises(AttributeError):
        ranking.candidates.append(ranking.candidates[0])  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="teacher_id"):
        TeacherRankingInput.create(
            example=record("rank-002", "Solve another task", "reference"),
            candidates=(
                TeacherCandidate.create("teacher-a", "first", 0.5),
                TeacherCandidate.create("teacher-a", "second", 0.6),
            ),
        )


def test_contamination_scan_blocks_normalized_overlap_with_benchmark_holdout() -> None:
    training = (
        record("train-001", "Write a robust parser", "implementation"),
        record("train-002", "Explain a tree", "explanation"),
    )
    benchmark = (
        record(
            "benchmark-001",
            " write   a robust parser ",
            "hidden reference",
            split=DatasetSplit.HOLDOUT,
        ),
    )

    report = scan_contamination(training_records=training, benchmark_records=benchmark)

    assert report.clean is False
    assert report.matches[0].training_record_id == "train-001"
    assert report.matches[0].benchmark_record_id == "benchmark-001"
    with pytest.raises(ContaminationError):
        report.require_clean()
