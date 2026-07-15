from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class PreferenceTokenizationError(ValueError):
    """Raised when a preference row cannot be split without losing tokens."""


def _messages(row: Mapping[str, object], key: str) -> list[dict[str, str]]:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        raise PreferenceTokenizationError(f"{key} must be a non-empty message list")
    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise PreferenceTokenizationError(f"{key} contains an invalid message")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise PreferenceTokenizationError(f"{key} contains an invalid role")
        if not isinstance(content, str) or not content.strip():
            raise PreferenceTokenizationError(f"{key} contains empty content")
        messages.append({"role": role.strip(), "content": content})
    return messages


def _token_ids(tokenizer: object, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
    try:
        encoded = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tokenize=True,
            return_dict=True,
            **kwargs,
        )
    except Exception as exc:
        raise PreferenceTokenizationError("chat template tokenization failed") from exc
    values = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise PreferenceTokenizationError("chat template returned invalid token ids")
    token_ids = list(values)
    if not token_ids or not all(isinstance(token_id, int) for token_id in token_ids):
        raise PreferenceTokenizationError("chat template returned invalid token ids")
    return token_ids


def _shared_prefix_length(*sequences: Sequence[int]) -> int:
    return next(
        (
            index
            for index, values in enumerate(zip(*sequences, strict=False))
            if len(set(values)) != 1
        ),
        min(len(sequence) for sequence in sequences),
    )


def tokenize_prefix_stable_preference(
    row: Mapping[str, object],
    *,
    tokenizer: object,
    max_length: int | None = None,
    truncation_mode: str = "keep_start",
) -> dict[str, list[int]]:
    """Tokenize a chat preference without assuming a stable text/token boundary."""

    prompt = _messages(row, "prompt")
    chosen = _messages(row, "chosen")
    rejected = _messages(row, "rejected")
    template_kwargs = row.get("chat_template_kwargs", {})
    if not isinstance(template_kwargs, Mapping):
        raise PreferenceTokenizationError("chat_template_kwargs must be a mapping")
    tools = row.get("tools")
    common_kwargs: dict[str, object] = dict(template_kwargs)
    if tools is not None:
        common_kwargs["tools"] = tools

    prompt_ids = _token_ids(
        tokenizer,
        prompt,
        add_generation_prompt=True,
        **common_kwargs,
    )
    chosen_full_ids = _token_ids(tokenizer, [*prompt, *chosen], **common_kwargs)
    rejected_full_ids = _token_ids(tokenizer, [*prompt, *rejected], **common_kwargs)
    boundary = _shared_prefix_length(prompt_ids, chosen_full_ids, rejected_full_ids)
    if boundary <= 0:
        raise PreferenceTokenizationError("preference variants have no shared token prefix")
    chosen_ids = chosen_full_ids[boundary:]
    rejected_ids = rejected_full_ids[boundary:]
    if not chosen_ids or not rejected_ids:
        raise PreferenceTokenizationError("preference completion produced no tokens")
    tokenized = {
        "prompt_ids": chosen_full_ids[:boundary],
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
    }
    return _truncate_preference(tokenized, max_length=max_length, mode=truncation_mode)


def _truncate_preference(
    tokenized: dict[str, list[int]], *, max_length: int | None, mode: str
) -> dict[str, list[int]]:
    if max_length is None:
        return tokenized
    if max_length < 2:
        raise PreferenceTokenizationError("max_length must leave room for prompt and completion")
    if mode not in {"keep_start", "keep_end"}:
        raise PreferenceTokenizationError("unsupported preference truncation mode")
    prompt = tokenized["prompt_ids"]
    prompt_length = min(len(prompt), max_length - 1)
    completion_length = max_length - prompt_length
    if mode == "keep_start":
        select_prompt = prompt[:prompt_length]
        select_chosen = tokenized["chosen_ids"][:completion_length]
        select_rejected = tokenized["rejected_ids"][:completion_length]
    else:
        select_prompt = prompt[-prompt_length:]
        select_chosen = tokenized["chosen_ids"][-completion_length:]
        select_rejected = tokenized["rejected_ids"][-completion_length:]
    if not select_chosen or not select_rejected:
        raise PreferenceTokenizationError("truncation removed all completion tokens")
    return {
        "prompt_ids": select_prompt,
        "chosen_ids": select_chosen,
        "rejected_ids": select_rejected,
    }


def prepare_prefix_stable_preference_dataset(
    dataset: Any,
    *,
    tokenizer: object,
    dataset_name: str,
    num_proc: int | None = None,
    max_length: int | None = None,
    truncation_mode: str = "keep_start",
) -> Any:
    """Add the exact token columns consumed by TRL's preference collator."""

    map_kwargs: dict[str, object] = {
        "desc": f"Prefix-stable tokenization of {dataset_name} dataset",
    }
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc
    return dataset.map(
        lambda row: tokenize_prefix_stable_preference(
            row,
            tokenizer=tokenizer,
            max_length=max_length,
            truncation_mode=truncation_mode,
        ),
        **map_kwargs,
    )


def build_prefix_stable_dpo_trainer(base_trainer: type) -> type:
    """Wrap the pinned TRL trainer while retaining its optimizer and loss implementation."""

    class PrefixStableDPOTrainer(base_trainer):
        def _prepare_dataset(
            self,
            dataset: Any,
            processing_class: object,
            args: object,
            dataset_name: str,
        ) -> Any:
            return prepare_prefix_stable_preference_dataset(
                dataset,
                tokenizer=processing_class,
                dataset_name=dataset_name,
                num_proc=getattr(args, "dataset_num_proc", None),
                max_length=getattr(args, "max_length", None),
                truncation_mode=getattr(args, "truncation_mode", "keep_start"),
            )

    PrefixStableDPOTrainer.__name__ = "PrefixStableDPOTrainer"
    PrefixStableDPOTrainer.__qualname__ = "PrefixStableDPOTrainer"
    return PrefixStableDPOTrainer
