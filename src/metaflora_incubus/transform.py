"""Pure tensor operations used by architecture adapters."""

from __future__ import annotations


def apply_directional_projection(weight, direction, *, strength: float):
    """Return a new weight matrix with part of one input direction removed.

    ``weight`` is never modified.  The operation is intentionally isolated so
    architecture-specific code only chooses *where* to apply it.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if weight.ndim != 2:
        raise ValueError("weight must be a two-dimensional matrix")
    if direction.ndim != 1 or direction.shape[0] != weight.shape[1]:
        raise ValueError("direction must match the matrix input dimension")

    normalized = direction.to(device=weight.device, dtype=weight.dtype)
    norm = normalized.norm()
    if norm.item() == 0:
        raise ValueError("direction must not be zero")
    unit_direction = normalized / norm
    component = weight @ unit_direction
    return weight - strength * component.unsqueeze(1) * unit_direction.unsqueeze(0)
