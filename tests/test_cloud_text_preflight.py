from __future__ import annotations

import json
from pathlib import Path

import pytest

from metaflora_incubus.cloud_training_runtime import (
    CloudConstraintError,
    preflight_training_text,
    require_benchmark_isolation,
)


class FakeTokenizer:
    chat_template = "safe-template"

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is True
        assert add_generation_prompt is False
        return [token for message in messages for token in message["content"].split()]


def write_jsonl(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def build_surfaces(root: Path, *, words: int = 2) -> None:
    text = " ".join("word" for _ in range(words))
    sft = (
        {"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": text}]},
    )
    preference = (
        {
            "prompt": [{"role": "user", "content": text}],
            "chosen": [{"role": "assistant", "content": text}],
            "rejected": [{"role": "assistant", "content": "other"}],
        },
    )
    write_jsonl(root / "sft.jsonl", sft)
    write_jsonl(root / "sft_validation.jsonl", sft)
    write_jsonl(root / "preference.jsonl", preference)
    write_jsonl(root / "preference_validation.jsonl", preference)


def test_text_preflight_checks_all_surfaces_before_model_loading(tmp_path: Path) -> None:
    build_surfaces(tmp_path)

    report = preflight_training_text(
        tokenizer=FakeTokenizer(),
        dataset_root=tmp_path,
        max_sequence_length=16,
    )

    assert report.checked_records == 4
    assert report.maximum_tokens == 4
    assert report.max_sequence_length == 16


def test_text_preflight_rejects_missing_template_and_overlong_or_malformed_rows(
    tmp_path: Path,
) -> None:
    build_surfaces(tmp_path, words=10)
    tokenizer = FakeTokenizer()
    with pytest.raises(CloudConstraintError, match="token length"):
        preflight_training_text(
            tokenizer=tokenizer,
            dataset_root=tmp_path,
            max_sequence_length=8,
        )

    tokenizer.chat_template = None
    with pytest.raises(CloudConstraintError, match="chat template"):
        preflight_training_text(
            tokenizer=tokenizer,
            dataset_root=tmp_path,
            max_sequence_length=64,
        )

    tokenizer.chat_template = "safe-template"
    write_jsonl(tmp_path / "preference.jsonl", ({"prompt": [], "chosen": []},))
    with pytest.raises(CloudConstraintError, match="schema"):
        preflight_training_text(
            tokenizer=tokenizer,
            dataset_root=tmp_path,
            max_sequence_length=64,
        )


def test_runtime_rejects_preference_prompt_from_release_benchmark(tmp_path: Path) -> None:
    build_surfaces(tmp_path)
    cases = tmp_path / "cases.jsonl"
    write_jsonl(cases, ({"case_id": "one", "prompt": "word word"},))

    with pytest.raises(CloudConstraintError, match="overlaps"):
        require_benchmark_isolation(dataset_root=tmp_path, cases_path=cases)
