"""Generic architecture discovery for arbitrary Transformer checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

TARGET_SUFFIXES = ("o_proj", "out_proj", "down_proj", "c_proj", "dense")


@dataclass(frozen=True)
class TransformTarget:
    module_name: str
    input_width: int
    output_width: int
    priority: int
    input_axis: int


def discover_transform_targets(
    named_modules: Iterable[tuple[str, object]],
) -> tuple[TransformTarget, ...]:
    """Find candidate projection matrices without a hard-coded model-family list."""
    targets: list[TransformTarget] = []
    for name, module in named_modules:
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            continue
        is_projection_like = (
            name.endswith(TARGET_SUFFIXES)
            or (hasattr(module, "in_features") and hasattr(module, "out_features"))
            or module.__class__.__name__ == "Conv1D"
        )
        if not is_projection_like:
            continue
        priority = 2 if name.endswith(("o_proj", "out_proj")) else 1
        is_conv1d = module.__class__.__name__ == "Conv1D"
        input_axis = 0 if is_conv1d else 1
        targets.append(
            TransformTarget(
                module_name=name,
                input_width=int(weight.shape[input_axis]),
                output_width=int(weight.shape[1 - input_axis]),
                priority=priority,
                input_axis=input_axis,
            )
        )
    return tuple(sorted(targets, key=lambda target: (-target.priority, target.module_name)))
