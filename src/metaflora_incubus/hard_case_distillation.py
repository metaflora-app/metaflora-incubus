"""Build deterministic private training patches from attested candidate failures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_PATCH_FILES = ("preference.jsonl", "provenance.jsonl", "sft.jsonl")
_SUPPORTED_DIMENSIONS = frozenset(
    {"agentic_search", "coding", "english", "russian", "safety", "tool_calling"}
)


class HardCaseDistillationError(ValueError):
    """Raised when hard-case patch inputs cannot be bound safely."""


@dataclass(frozen=True)
class HardCasePatchResult:
    output_dir: Path
    manifest_sha256: str
    candidate_artifact_sha256: str
    failure_count: int
    patch_count: int


def build_hard_case_patch(
    *,
    benchmark_cases: Path | str,
    benchmark_raw: Path | str,
    teacher_corpus: Path | str,
    output_dir: Path | str,
    private_output_root: Path | str,
    expected_cases_sha256: str,
    expected_raw_sha256: str,
    expected_teacher_sha256: str,
    prohibited_identifiers: Sequence[str],
    failure_score_threshold: float = 0.75,
) -> HardCasePatchResult:
    """Bind failed cases to transfer examples without training on held-out prompts."""

    if (
        not isinstance(failure_score_threshold, (int, float))
        or isinstance(failure_score_threshold, bool)
        or not 0.0 < float(failure_score_threshold) <= 1.0
    ):
        raise HardCaseDistillationError("failure score threshold is invalid")
    prohibited = tuple(
        normalized for item in prohibited_identifiers if (normalized := _normalize_text(item))
    )
    if not prohibited:
        raise HardCaseDistillationError("prohibited identifiers are required")

    cases_path = Path(benchmark_cases)
    raw_path = Path(benchmark_raw)
    teacher_path = Path(teacher_corpus)
    _verify_input(cases_path, expected_cases_sha256, "benchmark cases")
    _verify_input(raw_path, expected_raw_sha256, "benchmark raw output")
    _verify_input(teacher_path, expected_teacher_sha256, "teacher corpus")

    cases = _case_index(_read_jsonl(cases_path))
    raw, candidate_sha = _raw_index(_read_jsonl(raw_path), cases)
    failures = {
        case_id: row
        for case_id, row in raw.items()
        if bool(row["refused"]) or float(row["score"]) < float(failure_score_threshold)
    }
    if not failures:
        raise HardCaseDistillationError("candidate benchmark has no hard-case failures")
    teachers = _teacher_index(_read_jsonl(teacher_path), prohibited)
    if set(teachers) != set(failures):
        raise HardCaseDistillationError("teacher corpus must bind every failed case exactly")

    held_out_prompts = tuple(_normalize_text(str(case["prompt"])) for case in cases.values())
    sft_rows: list[dict[str, object]] = []
    preference_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    for case_id in sorted(failures):
        case = cases[case_id]
        failure = failures[case_id]
        teacher = teachers[case_id]
        source_prompt_sha = _sha256_bytes(str(case["prompt"]).encode())
        if teacher["source_prompt_sha256"] != source_prompt_sha:
            raise HardCaseDistillationError("teacher source prompt binding does not match")
        transfer_prompt = str(teacher["transfer_prompt"])
        if _crosses_contamination_boundary(transfer_prompt, held_out_prompts):
            raise HardCaseDistillationError("benchmark contamination boundary was crossed")
        chosen = str(teacher["chosen"])
        rejected = str(teacher["rejected"])
        patch_id = _sha256_bytes(
            _canonical_json(
                {
                    "candidate_artifact_sha256": candidate_sha,
                    "case_id": case_id,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source_record_id": teacher["source_record_id"],
                    "transfer_prompt": transfer_prompt,
                }
            )
        )
        capability = _capability(case)
        content_sha = _sha256_bytes(
            _canonical_json(
                {"chosen": chosen, "rejected": rejected, "transfer_prompt": transfer_prompt}
            )
        )
        sft_rows.append(
            {
                "capability": capability,
                "content_sha256": content_sha,
                "messages": [
                    {"content": transfer_prompt, "role": "user"},
                    {"content": chosen, "role": "assistant"},
                ],
                "patch_id": patch_id,
                "source_case_id": case_id,
            }
        )
        preference_rows.append(
            {
                "capability": capability,
                "chosen": [{"content": chosen, "role": "assistant"}],
                "content_sha256": content_sha,
                "patch_id": patch_id,
                "prompt": [{"content": transfer_prompt, "role": "user"}],
                "rejected": [{"content": rejected, "role": "assistant"}],
                "source_case_id": case_id,
            }
        )
        provenance_rows.append(
            {
                "benchmark_prompt_sha256": source_prompt_sha,
                "candidate_artifact_sha256": candidate_sha,
                "candidate_refused": bool(failure["refused"]),
                "candidate_score": float(failure["score"]),
                "license_id": teacher["license_id"],
                "patch_id": patch_id,
                "source_case_id": case_id,
                "source_record_id": teacher["source_record_id"],
                "source_revision": teacher["source_revision"],
                "teacher_corpus_sha256": expected_teacher_sha256,
                "transfer_prompt_sha256": _sha256_bytes(transfer_prompt.encode()),
            }
        )

    payloads = {
        "preference.jsonl": _jsonl_bytes(preference_rows),
        "provenance.jsonl": _jsonl_bytes(provenance_rows),
        "sft.jsonl": _jsonl_bytes(sft_rows),
    }
    file_hashes = {name: _sha256_bytes(payload) for name, payload in payloads.items()}
    manifest = {
        "benchmark_cases_sha256": expected_cases_sha256,
        "benchmark_raw_sha256": expected_raw_sha256,
        "candidate_artifact_sha256": candidate_sha,
        "failure_count": len(failures),
        "failure_score_threshold": float(failure_score_threshold),
        "files": file_hashes,
        "patch_count": len(sft_rows),
        "schema_version": 1,
        "teacher_corpus_sha256": expected_teacher_sha256,
    }
    manifest_payload = _canonical_json(manifest) + b"\n"
    destination = _write_private_patch(
        output_dir=Path(output_dir),
        private_root=Path(private_output_root),
        payloads={**payloads, "manifest.json": manifest_payload},
    )
    return HardCasePatchResult(
        output_dir=destination,
        manifest_sha256=_sha256_bytes(manifest_payload),
        candidate_artifact_sha256=candidate_sha,
        failure_count=len(failures),
        patch_count=len(sft_rows),
    )


def _verify_input(path: Path, expected_sha: str, label: str) -> None:
    if _HEX_64.fullmatch(expected_sha) is None or not path.is_file() or path.is_symlink():
        raise HardCaseDistillationError(f"{label} SHA-256 input is invalid")
    if _sha256_file(path) != expected_sha:
        raise HardCaseDistillationError(f"{label} SHA-256 mismatch")


def _case_index(rows: tuple[Mapping[str, object], ...]) -> dict[str, Mapping[str, object]]:
    cases: dict[str, Mapping[str, object]] = {}
    for row in rows:
        case_id = _required_text(row, "case_id", "benchmark case")
        if case_id in cases:
            raise HardCaseDistillationError("benchmark case IDs must be unique")
        _required_text(row, "prompt", case_id)
        dimension = _required_text(row, "dimension", case_id)
        if dimension not in _SUPPORTED_DIMENSIONS:
            raise HardCaseDistillationError("benchmark dimension is invalid")
        language = _required_text(row, "language", case_id)
        if language not in {"ru", "en"}:
            raise HardCaseDistillationError("benchmark language is invalid")
        cases[case_id] = row
    if not cases:
        raise HardCaseDistillationError("benchmark cases are empty")
    return cases


def _raw_index(
    rows: tuple[Mapping[str, object], ...], cases: Mapping[str, Mapping[str, object]]
) -> tuple[dict[str, Mapping[str, object]], str]:
    indexed: dict[str, Mapping[str, object]] = {}
    artifact_ids: set[str] = set()
    for row in rows:
        case_id = _required_text(row, "case_id", "benchmark raw output")
        if case_id in indexed or case_id not in cases:
            raise HardCaseDistillationError("benchmark raw case IDs are invalid")
        artifact = _required_text(row, "artifact_sha256", case_id)
        if _HEX_64.fullmatch(artifact) is None:
            raise HardCaseDistillationError("candidate artifact SHA-256 is invalid")
        artifact_ids.add(artifact)
        if (
            row.get("dimension") != cases[case_id]["dimension"]
            or row.get("language") != cases[case_id]["language"]
        ):
            raise HardCaseDistillationError("benchmark raw metadata does not match case bank")
        score = row.get("score")
        refused = row.get("refused")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0.0 <= float(score) <= 1.0
            or not isinstance(refused, bool)
        ):
            raise HardCaseDistillationError("benchmark raw score is invalid")
        indexed[case_id] = row
    if set(indexed) != set(cases):
        raise HardCaseDistillationError("benchmark raw output is incomplete")
    if len(artifact_ids) != 1:
        raise HardCaseDistillationError("benchmark raw output must bind one candidate")
    return indexed, next(iter(artifact_ids))


def _teacher_index(
    rows: tuple[Mapping[str, object], ...], prohibited: Sequence[str]
) -> dict[str, Mapping[str, object]]:
    teachers: dict[str, Mapping[str, object]] = {}
    for row in rows:
        case_id = _required_text(row, "case_id", "teacher record")
        if case_id in teachers:
            raise HardCaseDistillationError("teacher case IDs must be unique")
        for name in (
            "source_prompt_sha256",
            "transfer_prompt",
            "chosen",
            "rejected",
            "source_record_id",
            "source_revision",
            "license_id",
        ):
            _required_text(row, name, case_id)
        if (
            _HEX_64.fullmatch(str(row["source_prompt_sha256"])) is None
            or _HEX_40.fullmatch(str(row["source_revision"])) is None
        ):
            raise HardCaseDistillationError("teacher provenance binding is invalid")
        if str(row["chosen"]).strip() == str(row["rejected"]).strip():
            raise HardCaseDistillationError("teacher preference responses must differ")
        private_surface = _normalize_text(
            "\n".join(str(row[name]) for name in row if isinstance(row[name], str))
        )
        if any(identifier in private_surface for identifier in prohibited):
            raise HardCaseDistillationError("teacher record contains a prohibited identifier")
        teachers[case_id] = row
    return teachers


def _capability(case: Mapping[str, object]) -> str:
    dimension = str(case["dimension"])
    if dimension == "coding":
        return "code"
    if dimension in {"tool_calling", "agentic_search"}:
        return "agentic_tools"
    return "russian_text" if case["language"] == "ru" else "english_text"


def _write_private_patch(
    *, output_dir: Path, private_root: Path, payloads: Mapping[str, bytes]
) -> Path:
    if (
        not private_root.is_dir()
        or private_root.is_symlink()
        or private_root.stat().st_mode & 0o077
    ):
        raise HardCaseDistillationError("private output root is invalid")
    resolved_root = private_root.resolve()
    resolved_output = output_dir.resolve()
    if (
        resolved_root not in resolved_output.parents
        or output_dir.exists()
        or output_dir.is_symlink()
    ):
        raise HardCaseDistillationError("private patch output path is unsafe")
    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if staging.exists() or staging.is_symlink():
        raise HardCaseDistillationError("private patch staging path is unsafe")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(mode=0o700)
    try:
        for name in (*_PATCH_FILES, "manifest.json"):
            _write_exclusive(staging / name, payloads[name])
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise HardCaseDistillationError("cannot write private patch") from exc


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise HardCaseDistillationError(f"cannot read JSONL input: {path.name}") from exc
    rows: list[Mapping[str, object]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise HardCaseDistillationError(f"empty JSONL row: {path.name}:{number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HardCaseDistillationError(f"invalid JSONL row: {path.name}:{number}") from exc
        if not isinstance(row, Mapping):
            raise HardCaseDistillationError(f"JSONL row must be an object: {path.name}:{number}")
        rows.append(row)
    if not rows:
        raise HardCaseDistillationError(f"JSONL input is empty: {path.name}")
    return tuple(rows)


def _required_text(row: Mapping[str, object], name: str, context: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise HardCaseDistillationError(f"missing {name} in {context}")
    return value.strip()


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(re.findall(r"[a-zа-яё0-9]+", value.casefold()))


def _crosses_contamination_boundary(transfer_prompt: str, held_out_prompts: Sequence[str]) -> bool:
    normalized = _normalize_text(transfer_prompt)
    for held_out in held_out_prompts:
        if normalized == held_out:
            return True
        shorter, longer = sorted((normalized, held_out), key=len)
        if len(shorter) >= 32 and shorter in longer:
            return True
    return False


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise HardCaseDistillationError(f"cannot hash input: {path.name}") from exc
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
