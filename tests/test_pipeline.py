import json
from pathlib import Path

import pytest
import torch
from torch import nn

from metaflora_incubus.manifest import RunManifest
from metaflora_incubus.pipeline import (
    build_candidate,
    calibrate_input_directions,
    export_candidate,
    load_calibration_pairs,
)


class FakeTokenizer:
    def __call__(self, prompt: str, **_kwargs):
        start = max(len(prompt), 1)
        return {"input_ids": torch.tensor([[start, start + 1]])}

    def save_pretrained(self, output: Path) -> None:
        (output / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(2, 2, bias=False)
        nn.init.eye_(self.o_proj.weight)

    def forward(self, input_ids):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return self.o_proj(hidden)

    def save_pretrained(self, output: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        (output / "model.safetensors").write_bytes(b"weights")


def test_pipeline_creates_candidate_without_changing_source(tmp_path: Path) -> None:
    calibration_file = tmp_path / "calibration.jsonl"
    calibration_file.write_text(
        '{"baseline_prompt":"short","target_prompt":"a much longer target prompt"}\n',
        encoding="utf-8",
    )
    model = FakeModel()
    tokenizer = FakeTokenizer()
    pairs = load_calibration_pairs(calibration_file)

    targets, directions = calibrate_input_directions(model, tokenizer, pairs)
    candidate, changed = build_candidate(model, directions, strength=0.5)
    manifest = RunManifest.create(
        base_model="fake/model",
        model_revision="local",
        model_sha256="1" * 64,
        dataset_sha256="2" * 64,
        seed=42,
        strength=0.5,
        transform_version="0.1.0",
    )
    output = tmp_path / "incubus-v1"
    export_candidate(
        candidate=candidate,
        tokenizer=tokenizer,
        output_directory=output,
        manifest=manifest,
        changed_modules=changed,
    )

    assert targets[0].module_name == "o_proj"
    assert changed == ("o_proj",)
    assert torch.equal(model.o_proj.weight, torch.eye(2))
    assert not torch.equal(candidate.o_proj.weight, model.o_proj.weight)
    assert json.loads((output / "report.json").read_text())["source_model_preserved"] is True


def test_pipeline_rejects_invalid_or_empty_calibration(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("\n", encoding="utf-8")
    invalid_file = tmp_path / "invalid.jsonl"
    invalid_file.write_text('{"baseline_prompt":"only one field"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no usable"):
        load_calibration_pairs(empty_file)
    with pytest.raises(ValueError, match="needs baseline"):
        load_calibration_pairs(invalid_file)
