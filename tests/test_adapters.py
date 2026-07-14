from metaflora_incubus.adapters import discover_transform_targets


class FakeWeight:
    ndim = 2
    shape = (8, 4)


class FakeLinear:
    weight = FakeWeight()


class Conv1D:
    weight = FakeWeight()


def test_discovery_is_architecture_agnostic_and_prefers_attention_outputs() -> None:
    targets = discover_transform_targets(
        (
            ("model.layers.0.self_attn.o_proj", FakeLinear()),
            ("model.layers.0.mlp.down_proj", FakeLinear()),
            ("model.layers.0.self_attn.q_proj", FakeLinear()),
        )
    )

    assert [target.module_name for target in targets] == [
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
    ]


def test_discovery_uses_conv1d_input_orientation_for_c_proj() -> None:
    target = discover_transform_targets((("transformer.h.0.attn.c_proj", Conv1D()),))[0]

    assert target.input_axis == 0
    assert target.input_width == 8
