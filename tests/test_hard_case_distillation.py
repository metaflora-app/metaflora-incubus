from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from metaflora_incubus.hard_case_distillation import (
    HardCaseDistillationError,
    build_hard_case_patch,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    cases = root / "cases.jsonl"
    raw = root / "raw.jsonl"
    teacher = root / "teacher.jsonl"
    _write_jsonl(
        cases,
        [
            {
                "case_id": "coding-en-01",
                "dimension": "coding",
                "language": "en",
                "prompt": "Write a bounded integer helper in Python.",
            },
            {
                "case_id": "search-ru-01",
                "dimension": "agentic_search",
                "language": "ru",
                "prompt": "Найди два источника и сравни даты публикации.",
            },
            {
                "case_id": "english-en-01",
                "dimension": "english",
                "language": "en",
                "prompt": "Rewrite this paragraph clearly.",
            },
        ],
    )
    _write_jsonl(
        raw,
        [
            {
                "artifact_sha256": "a" * 64,
                "case_id": "coding-en-01",
                "dimension": "coding",
                "language": "en",
                "refused": False,
                "response": "weak",
                "score": 0.25,
            },
            {
                "artifact_sha256": "a" * 64,
                "case_id": "search-ru-01",
                "dimension": "agentic_search",
                "language": "ru",
                "refused": True,
                "response": "declined",
                "score": 0.0,
            },
            {
                "artifact_sha256": "a" * 64,
                "case_id": "english-en-01",
                "dimension": "english",
                "language": "en",
                "refused": False,
                "response": "clear response",
                "score": 1.0,
            },
        ],
    )
    case_rows = {
        row["case_id"]: row for row in [json.loads(line) for line in cases.read_text().splitlines()]
    }
    _write_jsonl(
        teacher,
        [
            {
                "case_id": "search-ru-01",
                "source_prompt_sha256": hashlib.sha256(
                    case_rows["search-ru-01"]["prompt"].encode()
                ).hexdigest(),
                "transfer_prompt": "Сопоставь даты в трёх предоставленных заметках.",
                "chosen": "Сначала фиксирую даты, затем сверяю их и указываю расхождения.",
                "rejected": "Я не могу сравнить эти заметки.",
                "source_record_id": "teacher-search-17",
                "source_revision": "b" * 40,
                "license_id": "Apache-2.0",
            },
            {
                "case_id": "coding-en-01",
                "source_prompt_sha256": hashlib.sha256(
                    case_rows["coding-en-01"]["prompt"].encode()
                ).hexdigest(),
                "transfer_prompt": "Implement a Python function that clamps a decimal value.",
                "chosen": (
                    "def clamp_decimal(value, low, high):\n    return max(low, min(value, high))"
                ),
                "rejected": "Use min or max somehow.",
                "source_record_id": "teacher-code-09",
                "source_revision": "b" * 40,
                "license_id": "Apache-2.0",
            },
        ],
    )
    return cases, raw, teacher


def _build(root: Path):
    cases, raw, teacher = _inputs(root)
    return build_hard_case_patch(
        benchmark_cases=cases,
        benchmark_raw=raw,
        teacher_corpus=teacher,
        output_dir=root / "private" / "patch",
        private_output_root=root / "private",
        expected_cases_sha256=_sha(cases),
        expected_raw_sha256=_sha(raw),
        expected_teacher_sha256=_sha(teacher),
        prohibited_identifiers=("hidden-teacher-name", "source-model-name"),
        failure_score_threshold=0.75,
    )


def test_builds_deterministic_verified_patch_only_for_failed_cases(tmp_path: Path) -> None:
    (tmp_path / "private").mkdir(mode=0o700)
    first = _build(tmp_path)
    first_payloads = {
        path.name: path.read_bytes() for path in first.output_dir.iterdir() if path.is_file()
    }
    other = tmp_path / "other"
    other.mkdir()
    (other / "private").mkdir(mode=0o700)
    second = _build(other)
    second_payloads = {
        path.name: path.read_bytes() for path in second.output_dir.iterdir() if path.is_file()
    }

    assert first.manifest_sha256 == second.manifest_sha256
    assert first_payloads == second_payloads
    assert first.failure_count == first.patch_count == 2
    assert set(first_payloads) == {
        "manifest.json",
        "preference.jsonl",
        "provenance.jsonl",
        "sft.jsonl",
    }
    sft = [json.loads(line) for line in first_payloads["sft.jsonl"].splitlines()]
    preference = [json.loads(line) for line in first_payloads["preference.jsonl"].splitlines()]
    assert {row["source_case_id"] for row in sft} == {"coding-en-01", "search-ru-01"}
    assert all(row["messages"][1]["content"] for row in sft)
    assert all(row["chosen"] != row["rejected"] for row in preference)
    assert "Write a bounded integer helper" not in first_payloads["sft.jsonl"].decode()
    assert all(path.stat().st_mode & 0o077 == 0 for path in first.output_dir.iterdir())
    with pytest.raises(FrozenInstanceError):
        first.patch_count = 3  # type: ignore[misc]


@pytest.mark.parametrize("failure", ("contamination", "prohibited", "missing_teacher"))
def test_rejects_unsafe_or_unbound_teacher_examples(tmp_path: Path, failure: str) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    cases, raw, teacher = _inputs(tmp_path)
    rows = [json.loads(line) for line in teacher.read_text().splitlines()]
    if failure == "contamination":
        rows[0]["transfer_prompt"] = "Найди два источника и сравни даты публикации."
    elif failure == "prohibited":
        rows[0]["chosen"] = "This contains hidden teacher name."
    else:
        rows = rows[:1]
    _write_jsonl(teacher, rows)

    with pytest.raises(HardCaseDistillationError):
        build_hard_case_patch(
            benchmark_cases=cases,
            benchmark_raw=raw,
            teacher_corpus=teacher,
            output_dir=private / "patch",
            private_output_root=private,
            expected_cases_sha256=_sha(cases),
            expected_raw_sha256=_sha(raw),
            expected_teacher_sha256=_sha(teacher),
            prohibited_identifiers=("hidden teacher name",),
            failure_score_threshold=0.75,
        )


def test_rejects_unverified_inputs_duplicate_rows_and_multiple_candidates(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    cases, raw, teacher = _inputs(tmp_path)

    with pytest.raises(HardCaseDistillationError, match="SHA-256"):
        build_hard_case_patch(
            benchmark_cases=cases,
            benchmark_raw=raw,
            teacher_corpus=teacher,
            output_dir=private / "bad-sha",
            private_output_root=private,
            expected_cases_sha256="0" * 64,
            expected_raw_sha256=_sha(raw),
            expected_teacher_sha256=_sha(teacher),
            prohibited_identifiers=("blocked",),
        )


def test_rejects_unknown_dimension_and_near_copy_contamination(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    cases, raw, teacher = _inputs(tmp_path)
    case_rows = [json.loads(line) for line in cases.read_text().splitlines()]
    case_rows[0]["dimension"] = "unreviewed_dimension"
    _write_jsonl(cases, case_rows)
    raw_rows = [json.loads(line) for line in raw.read_text().splitlines()]
    raw_rows[0]["dimension"] = "unreviewed_dimension"
    _write_jsonl(raw, raw_rows)
    with pytest.raises(HardCaseDistillationError, match="dimension"):
        build_hard_case_patch(
            benchmark_cases=cases,
            benchmark_raw=raw,
            teacher_corpus=teacher,
            output_dir=private / "unknown",
            private_output_root=private,
            expected_cases_sha256=_sha(cases),
            expected_raw_sha256=_sha(raw),
            expected_teacher_sha256=_sha(teacher),
            prohibited_identifiers=("blocked",),
        )

    cases, raw, teacher = _inputs(tmp_path)
    teacher_rows = [json.loads(line) for line in teacher.read_text().splitlines()]
    teacher_rows[0]["transfer_prompt"] = (
        "Найди два источника и сравни даты публикации. Ответ дай таблицей."
    )
    _write_jsonl(teacher, teacher_rows)
    with pytest.raises(HardCaseDistillationError, match="contamination"):
        build_hard_case_patch(
            benchmark_cases=cases,
            benchmark_raw=raw,
            teacher_corpus=teacher,
            output_dir=private / "near-copy",
            private_output_root=private,
            expected_cases_sha256=_sha(cases),
            expected_raw_sha256=_sha(raw),
            expected_teacher_sha256=_sha(teacher),
            prohibited_identifiers=("blocked",),
        )


def test_cli_emits_only_safe_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    cases, raw, teacher = _inputs(tmp_path)
    script = Path(__file__).parents[1] / "scripts" / "build_hard_case_patch.py"
    spec = importlib.util.spec_from_file_location("build_hard_case_patch_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--benchmark-cases",
            str(cases),
            "--benchmark-raw",
            str(raw),
            "--teacher-corpus",
            str(teacher),
            "--output-dir",
            str(private / "patch"),
            "--private-output-root",
            str(private),
            "--expected-cases-sha256",
            _sha(cases),
            "--expected-raw-sha256",
            _sha(raw),
            "--expected-teacher-sha256",
            _sha(teacher),
            "--prohibited-identifier",
            "hidden-teacher-name",
        ],
    )

    assert module.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "candidate_artifact_sha256": "a" * 64,
        "failure_count": 2,
        "manifest_sha256": summary["manifest_sha256"],
        "patch_count": 2,
    }
    assert len(summary["manifest_sha256"]) == 64

    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    rows[1]["artifact_sha256"] = "c" * 64
    _write_jsonl(raw, rows)
    with pytest.raises(HardCaseDistillationError, match="one candidate"):
        build_hard_case_patch(
            benchmark_cases=cases,
            benchmark_raw=raw,
            teacher_corpus=teacher,
            output_dir=private / "mixed",
            private_output_root=private,
            expected_cases_sha256=_sha(cases),
            expected_raw_sha256=_sha(raw),
            expected_teacher_sha256=_sha(teacher),
            prohibited_identifiers=("blocked",),
        )
