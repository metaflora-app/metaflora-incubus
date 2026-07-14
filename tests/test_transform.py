import pytest
import torch

from metaflora_incubus.transform import apply_directional_projection


def test_directional_projection_removes_the_selected_component_without_mutating_source() -> None:
    source = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    direction = torch.tensor([1.0, 0.0])

    transformed = apply_directional_projection(source, direction, strength=1.0)

    assert torch.equal(source, torch.tensor([[3.0, 4.0], [1.0, 2.0]]))
    assert torch.allclose(transformed[:, 0], torch.zeros(2))
    assert torch.allclose(transformed[:, 1], source[:, 1])


def test_directional_projection_rejects_invalid_strength() -> None:
    with pytest.raises(ValueError, match="strength"):
        apply_directional_projection(torch.eye(2), torch.tensor([1.0, 0.0]), strength=1.1)


def test_directional_projection_rejects_wrong_shapes_and_zero_direction() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        apply_directional_projection(torch.ones(2), torch.tensor([1.0]), strength=1.0)
    with pytest.raises(ValueError, match="input dimension"):
        apply_directional_projection(torch.eye(2), torch.tensor([1.0]), strength=1.0)
    with pytest.raises(ValueError, match="must not be zero"):
        apply_directional_projection(torch.eye(2), torch.zeros(2), strength=1.0)
