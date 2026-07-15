from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from metaflora_incubus.teacher_augmentation import (
    TeacherAugmentationError,
    augment_prepared_dataset,
)
from metaflora_incubus.training_entrypoints import prepare_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_raw_dataset(path: Path) -> None:
    rows = (
        {
            "record_id": f"existing-{capability}-{split}",
            "prompt": f"{capability} {split} prompt",
            "response": f"{capability} {split} response",
            "chosen": f"{capability} {split} preferred response",
            "rejected": f"{capability} {split} weak response",
            "capability": capability,
            "source_url": "https://datasets.example/original",
            "source_revision": "a" * 40,
            "collected_at": "2026-07-14T00:00:00Z",
            "license_id": "Apache-2.0",
            "split": split,
        }
        for capability in ("code", "agentic_tools", "russian_text", "english_text")
        for split in ("train", "validation")
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _prepared_dataset(tmp_path: Path) -> Path:
    raw = tmp_path / "raw.jsonl"
    prepared = tmp_path / "prepared"
    _write_raw_dataset(raw)
    prepare_dataset(
        input_path=raw,
        output_dir=prepared,
        expected_input_sha256=_sha256(raw),
    )
    return prepared


def _augmentation_integrity(tmp_path: Path, prepared: Path, teacher: Path) -> dict[str, object]:
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps({"case_id": "held-out", "prompt": "A genuinely held-out prompt"}) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((prepared / "manifest.json").read_text())
    return {
        "expected_prepared_sha256": manifest["dataset_sha256"],
        "expected_teacher_sha256": _sha256(teacher),
        "benchmark_cases": benchmark,
        "expected_benchmark_sha256": _sha256(benchmark),
        "private_output_root": tmp_path,
    }


def _write_teacher_dataset(path: Path) -> None:
    rows = [
        {
            "id": f"teacher-{index}",
            "input": f"Solve programming task {index}",
            "output": f"Complete verified solution {index}. " + "x" * 100,
            "domain": "coding" if index % 2 == 0 else "debugging",
        }
        for index in range(8)
    ]
    rows.append(
        {
            "id": "identity-leak",
            "input": "Who generated this record?",
            "output": "PrivateTeacher generated this response. " + "x" * 100,
            "domain": "coding",
        }
    )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + "{malformed teacher row\n",
        encoding="utf-8",
    )


def test_augmentation_reconstructs_existing_rows_and_is_deterministic(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    teacher = tmp_path / "teacher.jsonl"
    _write_teacher_dataset(teacher)
    integrity = _augmentation_integrity(tmp_path, prepared, teacher)

    first = augment_prepared_dataset(
        prepared_dataset=prepared,
        teacher_jsonl=teacher,
        output_path=tmp_path / "first.jsonl",
        source_url="https://datasets.example/teacher",
        source_revision="b" * 40,
        collected_at="2026-07-15T00:00:00Z",
        license_id="Apache-2.0",
        capability="code",
        train_count=3,
        validation_count=2,
        allowed_domains=("coding", "debugging"),
        prohibited_identifiers=("PrivateTeacher",),
        **integrity,
    )
    second = augment_prepared_dataset(
        prepared_dataset=prepared,
        teacher_jsonl=teacher,
        output_path=tmp_path / "second.jsonl",
        source_url="https://datasets.example/teacher",
        source_revision="b" * 40,
        collected_at="2026-07-15T00:00:00Z",
        license_id="Apache-2.0",
        capability="code",
        train_count=3,
        validation_count=2,
        allowed_domains=("debugging", "coding"),
        prohibited_identifiers=("privateteacher",),
        **integrity,
    )

    assert first.output_sha256 == second.output_sha256
    assert first.existing_count == 8
    assert first.added_train_count == 3
    assert first.added_validation_count == 2
    rows = [json.loads(line) for line in first.output_path.read_text().splitlines()]
    assert len(rows) == 13
    assert sum(row["source_url"].endswith("/teacher") for row in rows) == 5
    assert all(row["chosen"] != row["rejected"] for row in rows)
    assert "privateteacher" not in first.output_path.read_text().casefold()
    assert first.output_path.stat().st_mode & 0o077 == 0


def test_augmentation_rejects_too_small_teacher_corpus(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    teacher = tmp_path / "teacher.jsonl"
    _write_teacher_dataset(teacher)
    integrity = _augmentation_integrity(tmp_path, prepared, teacher)

    with pytest.raises(TeacherAugmentationError, match="eligible teacher records"):
        augment_prepared_dataset(
            prepared_dataset=prepared,
            teacher_jsonl=teacher,
            output_path=tmp_path / "output.jsonl",
            source_url="https://datasets.example/teacher",
            source_revision="b" * 40,
            collected_at="2026-07-15T00:00:00Z",
            license_id="Apache-2.0",
            capability="code",
            train_count=8,
            validation_count=2,
            allowed_domains=("coding", "debugging"),
            prohibited_identifiers=("PrivateTeacher",),
            **integrity,
        )


def test_augmentation_rejects_incomplete_prepared_bundle(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    teacher = tmp_path / "teacher.jsonl"
    _write_teacher_dataset(teacher)
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text('{"case_id":"one","prompt":"held out"}\n')

    with pytest.raises(TeacherAugmentationError, match="integrity verification"):
        augment_prepared_dataset(
            prepared_dataset=prepared,
            teacher_jsonl=teacher,
            output_path=tmp_path / "output.jsonl",
            source_url="https://datasets.example/teacher",
            source_revision="b" * 40,
            collected_at="2026-07-15T00:00:00Z",
            license_id="Apache-2.0",
            capability="code",
            train_count=1,
            validation_count=1,
            expected_prepared_sha256="0" * 64,
            expected_teacher_sha256=_sha256(teacher),
            benchmark_cases=benchmark,
            expected_benchmark_sha256=_sha256(benchmark),
            private_output_root=tmp_path,
            prohibited_identifiers=("PrivateTeacher",),
        )


def test_augmentation_rejects_benchmark_prompt_contamination(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    teacher = tmp_path / "teacher.jsonl"
    _write_teacher_dataset(teacher)
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        "".join(
            json.dumps({"case_id": f"leak-{index}", "prompt": f"Solve programming task {index}"})
            + "\n"
            for index in range(8)
        )
    )
    manifest = json.loads((prepared / "manifest.json").read_text())

    with pytest.raises(TeacherAugmentationError, match="benchmark contamination"):
        augment_prepared_dataset(
            prepared_dataset=prepared,
            teacher_jsonl=teacher,
            output_path=tmp_path / "output.jsonl",
            source_url="https://datasets.example/teacher",
            source_revision="b" * 40,
            collected_at="2026-07-15T00:00:00Z",
            license_id="Apache-2.0",
            capability="code",
            train_count=3,
            validation_count=2,
            prohibited_identifiers=("PrivateTeacher",),
            expected_prepared_sha256=manifest["dataset_sha256"],
            expected_teacher_sha256=_sha256(teacher),
            benchmark_cases=benchmark,
            expected_benchmark_sha256=_sha256(benchmark),
            private_output_root=tmp_path,
        )
