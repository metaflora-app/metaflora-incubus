from dataclasses import FrozenInstanceError, replace

import pytest
from metaflora_incubus.training_contract import (
    CheckpointCompatibilityError,
    CheckpointRef,
    DataMix,
    StageKind,
    TrainingConfig,
    TrainingPlan,
    TrainingStage,
    create_resume_plan,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def config(**overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "seed": 1701,
        "dataset_sha256": SHA_A,
        "source_artifact_sha256": SHA_B,
        "sequence_length": 32768,
        "effective_batch_size": 128,
        "learning_rate": 2e-5,
        "epochs": 2,
        "precision": "bf16",
    }
    values.update(overrides)
    return TrainingConfig.create(**values)


def stage(kind: StageKind, *, seed: int) -> TrainingStage:
    return TrainingStage.create(
        kind=kind,
        seed=seed,
        data_mix=DataMix(
            code=0.35,
            agentic_tools=0.25,
            russian_text=0.20,
            english_text=0.20,
        ),
        max_steps=100,
        checkpoint_every=20,
    )


def test_training_config_is_immutable_and_has_a_canonical_hash() -> None:
    first = config()
    reordered = TrainingConfig.create(
        precision="bf16",
        epochs=2,
        learning_rate=2e-5,
        effective_batch_size=128,
        sequence_length=32768,
        source_artifact_sha256=SHA_B,
        dataset_sha256=SHA_A,
        seed=1701,
    )

    assert first.config_sha256 == reordered.config_sha256
    assert len(first.config_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        first.seed = 9  # type: ignore[misc]
    assert replace(first, seed=9).config_sha256 == first.config_sha256
    assert config(seed=9).config_sha256 != first.config_sha256


@pytest.mark.parametrize(
    "override",
    (
        {"seed": -1},
        {"sequence_length": 0},
        {"effective_batch_size": 0},
        {"learning_rate": 0.0},
        {"epochs": 0},
        {"precision": "fp32"},
        {"dataset_sha256": "bad"},
    ),
)
def test_training_config_rejects_non_reproducible_values(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        config(**override)


def test_plan_requires_sft_then_preference_distillation() -> None:
    plan = TrainingPlan.create(
        config=config(),
        stages=(
            stage(StageKind.SFT, seed=1701),
            stage(StageKind.PREFERENCE_DISTILLATION, seed=1702),
        ),
    )

    assert tuple(item.kind for item in plan.stages) == (
        StageKind.SFT,
        StageKind.PREFERENCE_DISTILLATION,
    )
    assert (
        plan.plan_sha256
        == TrainingPlan.create(
            config=config(),
            stages=(
                stage(StageKind.SFT, seed=1701),
                stage(StageKind.PREFERENCE_DISTILLATION, seed=1702),
            ),
        ).plan_sha256
    )
    with pytest.raises(ValueError, match="SFT"):
        TrainingPlan.create(
            config=config(),
            stages=(stage(StageKind.PREFERENCE_DISTILLATION, seed=1702),),
        )
    with pytest.raises(ValueError, match="order"):
        TrainingPlan.create(
            config=config(),
            stages=(
                stage(StageKind.PREFERENCE_DISTILLATION, seed=1702),
                stage(StageKind.SFT, seed=1701),
            ),
        )


def test_data_mix_is_normalized_and_covers_product_capabilities() -> None:
    mix = stage(StageKind.SFT, seed=1701).data_mix

    assert sum((mix.code, mix.agentic_tools, mix.russian_text, mix.english_text)) == pytest.approx(
        1.0
    )
    with pytest.raises(ValueError, match="sum"):
        DataMix(code=0.5, agentic_tools=0.5, russian_text=0.5, english_text=0.5)
    with pytest.raises(ValueError, match="positive"):
        DataMix(code=0.5, agentic_tools=0.5, russian_text=0.0, english_text=0.0)


def test_checkpoint_resume_requires_exact_plan_dataset_and_source_artifact() -> None:
    plan = TrainingPlan.create(
        config=config(),
        stages=(
            stage(StageKind.SFT, seed=1701),
            stage(StageKind.PREFERENCE_DISTILLATION, seed=1702),
        ),
    )
    checkpoint = CheckpointRef.create(
        path="checkpoints/step-000020",
        stage=StageKind.SFT,
        completed_step=20,
        checkpoint_sha256=SHA_C,
        plan_sha256=plan.plan_sha256,
        config_sha256=plan.config.config_sha256,
        dataset_sha256=plan.config.dataset_sha256,
        source_artifact_sha256=plan.config.source_artifact_sha256,
    )

    resume = create_resume_plan(plan, checkpoint)

    assert resume.next_step == 21
    assert resume.stage is StageKind.SFT
    assert resume.checkpoint == checkpoint

    for field, changed in (
        ("plan_sha256", "d" * 64),
        ("config_sha256", "d" * 64),
        ("dataset_sha256", "d" * 64),
        ("source_artifact_sha256", "d" * 64),
    ):
        with pytest.raises(CheckpointCompatibilityError, match=field):
            create_resume_plan(plan, replace(checkpoint, **{field: changed}))


def test_checkpoint_cannot_resume_beyond_the_stage_boundary() -> None:
    plan = TrainingPlan.create(
        config=config(),
        stages=(
            stage(StageKind.SFT, seed=1701),
            stage(StageKind.PREFERENCE_DISTILLATION, seed=1702),
        ),
    )
    checkpoint = CheckpointRef.create(
        path="checkpoints/invalid",
        stage=StageKind.SFT,
        completed_step=101,
        checkpoint_sha256=SHA_C,
        plan_sha256=plan.plan_sha256,
        config_sha256=plan.config.config_sha256,
        dataset_sha256=plan.config.dataset_sha256,
        source_artifact_sha256=plan.config.source_artifact_sha256,
    )

    with pytest.raises(CheckpointCompatibilityError, match="completed_step"):
        create_resume_plan(plan, checkpoint)
