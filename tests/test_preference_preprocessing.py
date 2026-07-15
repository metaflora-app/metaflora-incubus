from __future__ import annotations

from typing import Any

import pytest

from metaflora_incubus.preference_preprocessing import (
    PreferenceTokenizationError,
    build_prefix_stable_dpo_trainer,
    prepare_prefix_stable_preference_dataset,
    tokenize_prefix_stable_preference,
)


class BoundaryChangingTokenizer:
    """Model a chat template whose final prompt token changes with the answer."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        **_kwargs: Any,
    ) -> list[int] | dict[str, list[int]]:
        assert tokenize is True
        if add_generation_prompt:
            token_ids = [11, 12, 90]
        elif messages[-1]["content"] == "preferred":
            token_ids = [11, 12, 91, 31, 2]
        else:
            token_ids = [11, 12, 92, 41, 2]
        return {"input_ids": token_ids} if return_dict else token_ids


class FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def map(self, function, **_kwargs):
        return FakeDataset([{**row, **function(row)} for row in self.rows])


def preference_row() -> dict[str, object]:
    return {
        "capability": "code",
        "prompt": [{"role": "user", "content": "request"}],
        "chosen": [{"role": "assistant", "content": "preferred"}],
        "rejected": [{"role": "assistant", "content": "weak"}],
    }


def test_tokenization_moves_unstable_boundary_tokens_into_completions() -> None:
    tokenized = tokenize_prefix_stable_preference(
        preference_row(),
        tokenizer=BoundaryChangingTokenizer(),
    )

    assert tokenized == {
        "prompt_ids": [11, 12],
        "chosen_ids": [91, 31, 2],
        "rejected_ids": [92, 41, 2],
    }
    assert tokenized["prompt_ids"] + tokenized["chosen_ids"] == [11, 12, 91, 31, 2]
    assert tokenized["prompt_ids"] + tokenized["rejected_ids"] == [11, 12, 92, 41, 2]


def test_dataset_preparation_preserves_metadata_and_adds_token_columns() -> None:
    prepared = prepare_prefix_stable_preference_dataset(
        FakeDataset([preference_row()]),
        tokenizer=BoundaryChangingTokenizer(),
        dataset_name="train",
    )

    assert prepared.rows[0]["capability"] == "code"
    assert prepared.rows[0]["prompt_ids"] == [11, 12]
    assert prepared.rows[0]["chosen_ids"] == [91, 31, 2]
    assert prepared.rows[0]["rejected_ids"] == [92, 41, 2]


def test_tokenization_rejects_rows_without_a_shared_prompt_prefix() -> None:
    class NoSharedPrefixTokenizer(BoundaryChangingTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("add_generation_prompt"):
                return [10]
            return [20, 30]

    with pytest.raises(PreferenceTokenizationError, match="shared token prefix"):
        tokenize_prefix_stable_preference(
            preference_row(),
            tokenizer=NoSharedPrefixTokenizer(),
        )


def test_tokenization_rejects_non_conversational_rows() -> None:
    row = {**preference_row(), "prompt": "request"}

    with pytest.raises(PreferenceTokenizationError, match="prompt"):
        tokenize_prefix_stable_preference(
            row,
            tokenizer=BoundaryChangingTokenizer(),
        )


def test_trainer_adapter_bypasses_upstream_conversational_retokenization() -> None:
    class UpstreamTrainer:
        def _prepare_dataset(self, *_args, **_kwargs):
            raise AssertionError("upstream conversational tokenizer must not run")

    trainer_type = build_prefix_stable_dpo_trainer(UpstreamTrainer)
    trainer = trainer_type()
    args = type(
        "Args",
        (),
        {"dataset_num_proc": None, "max_length": 512, "truncation_mode": "keep_start"},
    )()

    prepared = trainer._prepare_dataset(
        FakeDataset([preference_row()]),
        BoundaryChangingTokenizer(),
        args,
        "train",
    )

    assert prepared.rows[0]["prompt_ids"] == [11, 12]


def test_trainer_adapter_truncates_long_pairs_to_512_without_losing_labels() -> None:
    prompt_ids = list(range(300))
    chosen_ids = list(range(1_000, 1_500))
    rejected_ids = list(range(2_000, 2_500))

    class LongTokenizer(BoundaryChangingTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("add_generation_prompt"):
                values = prompt_ids
            elif messages[-1]["content"] == "preferred":
                values = [*prompt_ids, *chosen_ids]
            else:
                values = [*prompt_ids, *rejected_ids]
            return {"input_ids": values}

    class UpstreamTrainer:
        pass

    trainer = build_prefix_stable_dpo_trainer(UpstreamTrainer)()
    args = type(
        "Args",
        (),
        {"dataset_num_proc": None, "max_length": 512, "truncation_mode": "keep_start"},
    )()
    prepared = trainer._prepare_dataset(
        FakeDataset([preference_row()]),
        LongTokenizer(),
        args,
        "train",
    ).rows[0]

    assert len(prepared["prompt_ids"] + prepared["chosen_ids"]) == 512
    assert len(prepared["prompt_ids"] + prepared["rejected_ids"]) == 512
    assert prepared["prompt_ids"] == prompt_ids
    assert prepared["chosen_ids"] == chosen_ids[:212]
    assert prepared["rejected_ids"] == rejected_ids[:212]
    assert prepared["chosen_ids"] and prepared["rejected_ids"]
