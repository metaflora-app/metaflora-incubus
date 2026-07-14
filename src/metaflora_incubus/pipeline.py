"""Local, reproducible weight transformation pipeline for an Incubus v1 build."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metaflora_incubus.adapters import discover_transform_targets
from metaflora_incubus.manifest import RunManifest
from metaflora_incubus.transform import apply_directional_projection


@dataclass(frozen=True)
class CalibrationPair:
    baseline_prompt: str
    target_prompt: str


def load_calibration_pairs(path: Path) -> tuple[CalibrationPair, ...]:
    """Read a local JSONL calibration set without sending it anywhere."""
    pairs: list[CalibrationPair] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        baseline = item.get("baseline_prompt")
        target = item.get("target_prompt")
        if not isinstance(baseline, str) or not isinstance(target, str):
            raise ValueError(
                f"calibration line {line_number} needs baseline_prompt and target_prompt"
            )
        pairs.append(CalibrationPair(baseline_prompt=baseline, target_prompt=target))
    if not pairs:
        raise ValueError("calibration file contains no usable prompt pairs")
    return tuple(pairs)


def calibrate_input_directions(model: Any, tokenizer: Any, pairs: tuple[CalibrationPair, ...]):
    """Measure target-minus-baseline activation directions at editable modules.

    The model stays in inference mode.  Hooks capture module inputs only, then
    remove themselves before a transformed copy of the model is made.
    """
    import torch

    targets = discover_transform_targets(model.named_modules())
    modules = dict(model.named_modules())
    measurements: dict[str, dict[str, list[Any]]] = {
        target.module_name: {"baseline": [], "target": []} for target in targets
    }
    active_group = "baseline"
    hooks = []

    def make_hook(name: str):
        def hook(_module, inputs):
            hidden = inputs[0].detach()
            if hidden.ndim == 3:
                hidden = hidden.mean(dim=(0, 1))
            elif hidden.ndim == 2:
                hidden = hidden.mean(dim=0)
            measurements[name][active_group].append(hidden.to(dtype=torch.float32).cpu())

        return hook

    for target in targets:
        hooks.append(
            modules[target.module_name].register_forward_pre_hook(make_hook(target.module_name))
        )

    device = next(model.parameters()).device
    model.eval()
    try:
        with torch.inference_mode():
            for pair in pairs:
                prompts = (("baseline", pair.baseline_prompt), ("target", pair.target_prompt))
                for group, prompt in prompts:
                    active_group = group
                    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
                    model(**{key: value.to(device) for key, value in encoded.items()})
    finally:
        for hook in hooks:
            hook.remove()

    directions = {}
    for target in targets:
        values = measurements[target.module_name]
        if not values["baseline"] or not values["target"]:
            continue
        direction = torch.stack(values["target"]).mean(0) - torch.stack(values["baseline"]).mean(0)
        if direction.numel() == target.input_width and direction.norm().item() > 0:
            directions[target.module_name] = direction
    return targets, directions


def build_candidate(model: Any, directions: dict[str, Any], *, strength: float):
    """Create a changed model copy while preserving the loaded build input."""
    import torch

    candidate = copy.deepcopy(model)
    modules = dict(candidate.named_modules())
    changed: list[str] = []
    for name, direction in directions.items():
        module = modules[name]
        original_weight = module.weight
        matrix = original_weight.detach()
        projection_matrix = matrix if direction.numel() == matrix.shape[1] else matrix.T
        projected = apply_directional_projection(projection_matrix, direction, strength=strength)
        if projection_matrix is not matrix:
            projected = projected.T
        module.weight = torch.nn.Parameter(projected, requires_grad=original_weight.requires_grad)
        changed.append(name)
    return candidate, tuple(changed)


def export_candidate(
    *,
    candidate: Any,
    tokenizer: Any,
    output_directory: Path,
    manifest: RunManifest,
    changed_modules: tuple[str, ...],
) -> None:
    """Write a new safe-serialization artifact and provenance record."""
    output_directory.mkdir(parents=True, exist_ok=False)
    candidate.save_pretrained(output_directory, safe_serialization=True)
    tokenizer.save_pretrained(output_directory)
    (output_directory / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    report = {
        "release": "Metaflora Incubus v1",
        "changed_modules": list(changed_modules),
        "source_model_preserved": True,
        "model_sha256": manifest.model_sha256,
        "dataset_sha256": manifest.dataset_sha256,
    }
    (output_directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
