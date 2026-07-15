"""Deterministically merge an external teacher corpus into a prepared private dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_PREPARED_FILES = (
    "sft.jsonl",
    "preference.jsonl",
    "sft_validation.jsonl",
    "preference_validation.jsonl",
    "provenance.jsonl",
)


class TeacherAugmentationError(ValueError):
    """Raised when an augmentation input cannot produce a safe deterministic merge."""


@dataclass(frozen=True)
class TeacherAugmentationResult:
    output_path: Path
    output_sha256: str
    existing_count: int
    added_train_count: int
    added_validation_count: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise TeacherAugmentationError(f"cannot hash input file: {path.name}") from exc
    return digest.hexdigest()


def _normalized_text(value: str) -> str:
    return "".join(re.findall(r"[a-zа-яё0-9]+", value.casefold()))


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _read_jsonl(path: Path, *, skip_malformed: bool = False) -> tuple[Mapping[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TeacherAugmentationError(f"cannot read JSONL file: {path.name}") from exc
    records: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if skip_malformed:
                continue
            raise TeacherAugmentationError(
                f"invalid JSON in {path.name} at line {line_number}"
            ) from exc
        if not isinstance(record, Mapping):
            raise TeacherAugmentationError(f"invalid record in {path.name} at line {line_number}")
        records.append(record)
    return tuple(records)


def _required_string(record: Mapping[str, object], key: str, *, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TeacherAugmentationError(f"missing {key} in {context}")
    return value.strip()


def _message_content(value: object, *, context: str, role: str) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TeacherAugmentationError(f"invalid messages in {context}")
    for message in value:
        if not isinstance(message, Mapping) or message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    raise TeacherAugmentationError(f"missing {role} message in {context}")


def _index_by_record_id(
    records: Iterable[Mapping[str, object]], *, context: str
) -> Mapping[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for record in records:
        record_id = _required_string(record, "record_id", context=context)
        if record_id in indexed:
            raise TeacherAugmentationError(f"duplicate record_id in {context}: {record_id}")
        indexed[record_id] = record
    return indexed


def _reconstruct_prepared_records(
    root: Path,
    *,
    expected_dataset_sha256: str,
    prohibited_identifiers: Sequence[str],
) -> tuple[dict[str, object], ...]:
    from metaflora_incubus.training_entrypoints import _load_dataset_manifest

    try:
        _load_dataset_manifest(root / "manifest.json", expected_dataset_sha256)
    except ValueError as exc:
        raise TeacherAugmentationError("prepared dataset integrity verification failed") from exc
    prohibited = tuple(
        _normalized_text(identifier)
        for identifier in prohibited_identifiers
        if _normalized_text(identifier)
    )
    for name in _PREPARED_FILES:
        if not (root / name).is_file():
            raise TeacherAugmentationError(f"prepared dataset file is missing: {name}")
    provenance = _index_by_record_id(_read_jsonl(root / "provenance.jsonl"), context="provenance")
    reconstructed: list[dict[str, object]] = []
    prepared_ids: set[str] = set()
    for split, sft_name, preference_name in (
        ("train", "sft.jsonl", "preference.jsonl"),
        ("validation", "sft_validation.jsonl", "preference_validation.jsonl"),
    ):
        sft = _index_by_record_id(_read_jsonl(root / sft_name), context=sft_name)
        preference = _index_by_record_id(
            _read_jsonl(root / preference_name), context=preference_name
        )
        if set(sft) != set(preference):
            raise TeacherAugmentationError(
                f"record mismatch between {sft_name} and {preference_name}"
            )
        for record_id in sorted(sft):
            prepared_ids.add(record_id)
            if record_id not in provenance:
                raise TeacherAugmentationError(f"missing provenance for record_id: {record_id}")
            sft_record = sft[record_id]
            preference_record = preference[record_id]
            provenance_record = provenance[record_id]
            prompt = _message_content(
                sft_record.get("messages"), context=f"{sft_name}:{record_id}", role="user"
            )
            preference_prompt = _message_content(
                preference_record.get("prompt"),
                context=f"{preference_name}:{record_id}",
                role="user",
            )
            if prompt != preference_prompt:
                raise TeacherAugmentationError(f"prompt mismatch for record_id: {record_id}")
            candidate = {
                "record_id": record_id,
                "prompt": prompt,
                "response": _message_content(
                    sft_record.get("messages"),
                    context=f"{sft_name}:{record_id}",
                    role="assistant",
                ),
                "chosen": _message_content(
                    preference_record.get("chosen"),
                    context=f"{preference_name}:{record_id}",
                    role="assistant",
                ),
                "rejected": _message_content(
                    preference_record.get("rejected"),
                    context=f"{preference_name}:{record_id}",
                    role="assistant",
                ),
                "capability": _required_string(
                    provenance_record, "capability", context=f"provenance:{record_id}"
                ),
                "source_url": _required_string(
                    provenance_record, "source_url", context=f"provenance:{record_id}"
                ),
                "source_revision": _required_string(
                    provenance_record,
                    "source_revision",
                    context=f"provenance:{record_id}",
                ),
                "collected_at": _required_string(
                    provenance_record, "collected_at", context=f"provenance:{record_id}"
                ),
                "license_id": _required_string(
                    provenance_record, "license_id", context=f"provenance:{record_id}"
                ),
                "split": split,
            }
            private_text = _normalized_text(
                "\n".join(
                    str(candidate[key]) for key in ("prompt", "response", "chosen", "rejected")
                )
            )
            if any(identifier in private_text for identifier in prohibited):
                continue
            reconstructed.append(candidate)
    if set(provenance) != prepared_ids:
        raise TeacherAugmentationError("orphan provenance records are forbidden")
    return tuple(reconstructed)


def _weaken_response(response: str) -> str:
    cutoff = max(24, min(len(response) // 5, 256))
    weakened = response[:cutoff].rstrip()
    suffix = " [incomplete]"
    candidate = weakened + suffix
    return candidate if candidate != response else "Incomplete solution."


def _eligible_teacher_records(
    path: Path,
    *,
    allowed_domains: Sequence[str],
    minimum_output_chars: int,
    maximum_output_chars: int,
    prohibited_identifiers: Sequence[str],
) -> tuple[tuple[str, str, str], ...]:
    allowed = frozenset(domain.strip().casefold() for domain in allowed_domains if domain.strip())
    prohibited = tuple(
        _normalized_text(identifier)
        for identifier in prohibited_identifiers
        if _normalized_text(identifier)
    )
    eligible: dict[str, tuple[str, str, str]] = {}
    for record in _read_jsonl(path, skip_malformed=True):
        domain_value = record.get("domain", "")
        domain = domain_value.strip().casefold() if isinstance(domain_value, str) else ""
        if allowed and domain not in allowed:
            continue
        prompt = record.get("input")
        response = record.get("output")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if not isinstance(response, str) or not response.strip():
            continue
        prompt = prompt.strip()
        response = response.strip()
        combined_text = _normalized_text(f"{prompt}\n{response}")
        if any(identifier in combined_text for identifier in prohibited):
            continue
        if not minimum_output_chars <= len(response) <= maximum_output_chars:
            continue
        identity = _sha256_bytes(_canonical_json({"prompt": prompt, "response": response}).encode())
        eligible[identity] = (identity, prompt, response)
    return tuple(eligible[key] for key in sorted(eligible))


def augment_prepared_dataset(
    *,
    prepared_dataset: Path | str,
    teacher_jsonl: Path | str,
    output_path: Path | str,
    source_url: str,
    source_revision: str,
    collected_at: str,
    license_id: str,
    capability: str,
    train_count: int,
    validation_count: int,
    allowed_domains: Sequence[str] = (),
    minimum_output_chars: int = 80,
    maximum_output_chars: int = 16_000,
    prohibited_identifiers: Sequence[str] = (),
    expected_prepared_sha256: str,
    expected_teacher_sha256: str,
    benchmark_cases: Path | str,
    expected_benchmark_sha256: str,
    private_output_root: Path | str,
) -> TeacherAugmentationResult:
    """Reconstruct a prepared bundle and add a deterministic teacher sample."""

    if train_count < 1 or validation_count < 1:
        raise TeacherAugmentationError("train and validation counts must be positive")
    if not prohibited_identifiers:
        raise TeacherAugmentationError("prohibited identifiers are required")
    revision_is_hex = all(char in "0123456789abcdef" for char in source_revision)
    if len(source_revision) != 40 or not revision_is_hex:
        raise TeacherAugmentationError("source revision must be a lowercase 40-character hash")
    if not 1 <= minimum_output_chars <= maximum_output_chars:
        raise TeacherAugmentationError("invalid teacher output length bounds")
    teacher_path = Path(teacher_jsonl)
    if _sha256_file(teacher_path) != expected_teacher_sha256:
        raise TeacherAugmentationError("teacher input sha256 mismatch")
    benchmark_path = Path(benchmark_cases)
    if _sha256_file(benchmark_path) != expected_benchmark_sha256:
        raise TeacherAugmentationError("benchmark cases sha256 mismatch")
    existing = _reconstruct_prepared_records(
        Path(prepared_dataset),
        expected_dataset_sha256=expected_prepared_sha256,
        prohibited_identifiers=prohibited_identifiers,
    )
    eligible = _eligible_teacher_records(
        teacher_path,
        allowed_domains=allowed_domains,
        minimum_output_chars=minimum_output_chars,
        maximum_output_chars=maximum_output_chars,
        prohibited_identifiers=prohibited_identifiers,
    )
    required = train_count + validation_count
    if len(eligible) < required:
        raise TeacherAugmentationError(
            f"not enough eligible teacher records: need {required}, found {len(eligible)}"
        )
    additions: list[dict[str, object]] = []
    for index, (identity, prompt, response) in enumerate(eligible[:required]):
        additions.append(
            {
                "record_id": f"external-{identity[:32]}",
                "prompt": prompt,
                "response": response,
                "chosen": response,
                "rejected": _weaken_response(response),
                "capability": capability,
                "source_url": source_url,
                "source_revision": source_revision,
                "collected_at": collected_at,
                "license_id": license_id,
                "split": "train" if index < train_count else "validation",
            }
        )
    combined = sorted((*existing, *additions), key=lambda record: str(record["record_id"]))
    held_out_prompts = {
        _normalized_text(_required_string(row, "prompt", context="benchmark"))
        for row in _read_jsonl(benchmark_path)
    }
    if any(_normalized_text(str(record["prompt"])) in held_out_prompts for record in combined):
        raise TeacherAugmentationError("benchmark contamination detected")
    payload = "".join(_canonical_json(record) + "\n" for record in combined).encode()
    target = Path(output_path)
    private_root = Path(private_output_root)
    if not private_root.is_dir() or private_root.is_symlink():
        raise TeacherAugmentationError("private output root is invalid")
    resolved_root = private_root.resolve()
    resolved_target = target.resolve()
    if resolved_root not in resolved_target.parents or target.is_symlink():
        raise TeacherAugmentationError("output must remain inside the private output root")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise TeacherAugmentationError("temporary output path is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise TeacherAugmentationError("cannot write private augmentation output") from exc
    temporary.replace(target)
    target.chmod(0o600)
    return TeacherAugmentationResult(
        output_path=target,
        output_sha256=_sha256_bytes(payload),
        existing_count=len(existing),
        added_train_count=train_count,
        added_validation_count=validation_count,
    )
