from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from metaflora_incubus.training_entrypoints import (
    CandidateState,
    HashMismatchError,
    TrainingInputError,
    build_training_run,
    calculate_source_artifact_sha256,
    load_maintainer_config,
    pending_stage_recipes,
    prepare_dataset,
)

CONFIG_PATH = Path("configs/training/incubus-v1.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source(path: Path, *, split: str = "train") -> None:
    records = tuple(
        {
            "record_id": f"{capability}-{record_split}",
            "prompt": f"{record_split} prompt for {capability}",
            "response": f"{record_split} answer for {capability}",
            "chosen": f"preferred {record_split} answer for {capability}",
            "rejected": f"inferior {record_split} answer for {capability}",
            "capability": capability,
            "source_url": f"https://datasets.example/{capability}/{record_split}",
            "source_revision": "b" * 40,
            "collected_at": "2026-07-14T12:00:00Z",
            "license_id": "Apache-2.0" if record_split == "train" else "MIT",
            "split": split if record_split == "train" else "validation",
        }
        for capability in ("code", "agentic_tools", "russian_text", "english_text")
        for record_split in ("train", "validation")
    )
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def prepare_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw.jsonl"
    prepared = tmp_path / "prepared"
    source = tmp_path / "source"
    write_source(raw)
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"safe local source artifact")
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    result = prepare_dataset(
        input_path=raw,
        output_dir=prepared,
        expected_input_sha256=sha256(raw),
    )
    return prepared / "manifest.json", source, result.manifest_path


def run_environment(config_path: Path, manifest: Path, source: Path) -> dict[str, str]:
    config = load_maintainer_config(config_path)
    manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        config.config_sha256_env: config.document_sha256,
        config.source.path_env: str(source),
        config.source.sha256_env: calculate_source_artifact_sha256(source),
        config.dataset.manifest_env: str(manifest),
        config.dataset.sha256_env: manifest_document["dataset_sha256"],
    }


def test_committed_config_is_brand_only_and_defines_both_training_stages() -> None:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    config = load_maintainer_config(CONFIG_PATH)

    assert config.product_id == "metaflora-incubus-v1"
    assert tuple(stage.kind.value for stage in config.stages) == (
        "sft",
        "preference_distillation",
    )
    assert all(
        name not in raw.casefold()
        for name in ("forbidden-build-input", "forbidden-teacher-a", "forbidden-teacher-b")
    )
    assert len(config.document_sha256) == 64


def test_dataset_preparation_is_deterministic_auditable_and_excludes_holdout(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.jsonl"
    write_source(raw)

    first = prepare_dataset(
        input_path=raw,
        output_dir=tmp_path / "first",
        expected_input_sha256=sha256(raw),
    )
    second = prepare_dataset(
        input_path=raw,
        output_dir=tmp_path / "second",
        expected_input_sha256=sha256(raw),
    )

    assert first.dataset_sha256 == second.dataset_sha256
    assert first.record_counts == {
        "preference": 4,
        "preference_validation": 4,
        "sft": 4,
        "sft_validation": 4,
    }
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert "holdout" not in first.manifest_path.read_text(encoding="utf-8").casefold()
    sft_row = json.loads((first.manifest_path.parent / "sft.jsonl").read_text().splitlines()[0])
    preference_row = json.loads(
        (first.manifest_path.parent / "preference.jsonl").read_text().splitlines()[0]
    )
    assert [message["role"] for message in sft_row["messages"]] == ["user", "assistant"]
    assert preference_row["prompt"][0]["role"] == "user"
    assert preference_row["chosen"][0]["role"] == "assistant"
    assert preference_row["rejected"][0]["role"] == "assistant"
    provenance_row = json.loads(
        (first.manifest_path.parent / "provenance.jsonl").read_text().splitlines()[0]
    )
    assert provenance_row["source_url"].startswith("https://datasets.example/")
    assert provenance_row["source_revision"] == "b" * 40
    assert provenance_row["license_id"] in {"Apache-2.0", "MIT"}

    write_source(raw, split="holdout")
    with pytest.raises(TrainingInputError, match="holdout"):
        prepare_dataset(
            input_path=raw,
            output_dir=tmp_path / "blocked",
            expected_input_sha256=sha256(raw),
        )


def test_dataset_preparation_fails_closed_on_input_hash_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    write_source(raw)

    with pytest.raises(HashMismatchError, match="input"):
        prepare_dataset(
            input_path=raw,
            output_dir=tmp_path / "prepared",
            expected_input_sha256="0" * 64,
        )


def test_dataset_preparation_enforces_license_dedup_and_split_leakage(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    write_source(raw)
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]

    unlicensed = [{**rows[0], "license_id": "unknown"}, *rows[1:]]
    raw.write_text("".join(json.dumps(row) + "\n" for row in unlicensed), encoding="utf-8")
    with pytest.raises(TrainingInputError, match="license"):
        prepare_dataset(
            input_path=raw,
            output_dir=tmp_path / "unlicensed",
            expected_input_sha256=sha256(raw),
        )

    duplicate = [*rows, {**rows[0], "record_id": "different-id"}]
    raw.write_text("".join(json.dumps(row) + "\n" for row in duplicate), encoding="utf-8")
    with pytest.raises(TrainingInputError, match="duplicate"):
        prepare_dataset(
            input_path=raw,
            output_dir=tmp_path / "duplicate",
            expected_input_sha256=sha256(raw),
        )

    leaked = [*rows]
    leaked[1] = {**leaked[1], "prompt": rows[0]["prompt"]}
    raw.write_text("".join(json.dumps(row) + "\n" for row in leaked), encoding="utf-8")
    with pytest.raises(TrainingInputError, match="leakage"):
        prepare_dataset(
            input_path=raw,
            output_dir=tmp_path / "leaked",
            expected_input_sha256=sha256(raw),
        )


def test_training_dry_run_needs_no_ml_dependencies_or_gpu_and_is_safe(tmp_path: Path) -> None:
    manifest, source, _ = prepare_inputs(tmp_path)
    config = load_maintainer_config(CONFIG_PATH)

    run = build_training_run(
        config_path=CONFIG_PATH,
        environment=run_environment(CONFIG_PATH, manifest, source),
        dry_run=True,
    )

    assert run.dry_run is True
    assert (
        run.plan.config.dataset_sha256
        == json.loads(manifest.read_text(encoding="utf-8"))["dataset_sha256"]
    )
    assert run.model_load_kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    assert run.stage_recipes[0].trainer == "SFTTrainer"
    assert run.stage_recipes[1].trainer == "DPOTrainer"
    assert all(
        recipe.seed == stage.seed
        for recipe, stage in zip(run.stage_recipes, config.stages, strict=True)
    )
    assert run.resume_checkpoint is None
    assert run.candidate_state is CandidateState.ADAPTER_ONLY
    assert run.release_ready is False
    assert run.post_training.steps == (
        "merge_adapter_safetensors",
        "export_gguf",
        "quantize_q4",
        "run_parity_and_release_gates",
    )
    assert all(
        recipe.data_mix == stage.data_mix
        for recipe, stage in zip(run.stage_recipes, config.stages, strict=True)
    )


def test_training_requires_a_loadable_local_safetensors_directory(tmp_path: Path) -> None:
    manifest, _source, _ = prepare_inputs(tmp_path)
    single_file = tmp_path / "single.safetensors"
    single_file.write_bytes(b"not a complete local model directory")
    environment = run_environment(CONFIG_PATH, manifest, single_file)

    with pytest.raises(TrainingInputError, match="directory"):
        build_training_run(
            config_path=CONFIG_PATH,
            environment=environment,
            dry_run=True,
        )


def test_training_checks_config_dataset_and_source_hashes_before_execution(tmp_path: Path) -> None:
    manifest, source, _ = prepare_inputs(tmp_path)
    environment = run_environment(CONFIG_PATH, manifest, source)
    config = load_maintainer_config(CONFIG_PATH)

    for variable in (
        config.config_sha256_env,
        config.dataset.sha256_env,
        config.source.sha256_env,
    ):
        changed = {**environment, variable: "f" * 64}
        with pytest.raises(HashMismatchError):
            build_training_run(
                config_path=CONFIG_PATH,
                environment=changed,
                dry_run=True,
            )


def test_execution_hash_covers_lora_and_per_device_batch_settings(tmp_path: Path) -> None:
    manifest, source, _ = prepare_inputs(tmp_path)
    base_environment = run_environment(CONFIG_PATH, manifest, source)
    base = build_training_run(
        config_path=CONFIG_PATH,
        environment=base_environment,
        dry_run=True,
    )
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    document["lora"]["rank"] = 32
    document["per_device_train_batch_size"] = 2
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(document), encoding="utf-8")
    changed_config = load_maintainer_config(changed_path)
    changed_environment = {
        **base_environment,
        changed_config.config_sha256_env: changed_config.document_sha256,
    }
    changed = build_training_run(
        config_path=changed_path,
        environment=changed_environment,
        dry_run=True,
    )

    assert changed.execution_config_sha256 != base.execution_config_sha256
    assert changed.execution_plan_sha256 != base.execution_plan_sha256


def test_resume_metadata_must_match_the_exact_plan_and_provenance(tmp_path: Path) -> None:
    manifest, source, _ = prepare_inputs(tmp_path)
    environment = run_environment(CONFIG_PATH, manifest, source)
    initial = build_training_run(
        config_path=CONFIG_PATH,
        environment=environment,
        dry_run=True,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "step-000020"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint")
    checkpoint_hash = calculate_source_artifact_sha256(checkpoint)
    metadata = {
        "path": "step-000020",
        "stage": "sft",
        "completed_step": 20,
        "checkpoint_sha256": checkpoint_hash,
        "plan_sha256": initial.plan.plan_sha256,
        "config_sha256": initial.plan.config.config_sha256,
        "dataset_sha256": initial.plan.config.dataset_sha256,
        "source_artifact_sha256": initial.plan.config.source_artifact_sha256,
        "execution_config_sha256": initial.execution_config_sha256,
        "execution_plan_sha256": initial.execution_plan_sha256,
    }
    resume_path = checkpoint_dir / "checkpoint.json"
    resume_path.write_text(json.dumps(metadata), encoding="utf-8")

    resumed = build_training_run(
        config_path=CONFIG_PATH,
        environment=environment,
        resume_metadata_path=resume_path,
        dry_run=True,
    )
    assert resumed.resume_checkpoint is not None
    assert resumed.resume_checkpoint.next_step == 21
    assert resumed.resume_path == checkpoint

    resume_path.write_text(json.dumps({**metadata, "dataset_sha256": "d" * 64}), encoding="utf-8")
    with pytest.raises(HashMismatchError, match="resume"):
        build_training_run(
            config_path=CONFIG_PATH,
            environment=environment,
            resume_metadata_path=resume_path,
            dry_run=True,
        )

    resume_path.write_text(json.dumps(metadata), encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(HashMismatchError, match="checkpoint"):
        build_training_run(
            config_path=CONFIG_PATH,
            environment=environment,
            resume_metadata_path=resume_path,
            dry_run=True,
        )


def test_preference_resume_does_not_rerun_sft(tmp_path: Path) -> None:
    manifest, source, _ = prepare_inputs(tmp_path)
    environment = run_environment(CONFIG_PATH, manifest, source)
    initial = build_training_run(
        config_path=CONFIG_PATH,
        environment=environment,
        dry_run=True,
    )
    checkpoint_root = tmp_path / "preference-checkpoint"
    checkpoint_root.mkdir()
    checkpoint = checkpoint_root / "step-000010"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"preference checkpoint")
    metadata = {
        "path": checkpoint.name,
        "stage": "preference_distillation",
        "completed_step": 10,
        "checkpoint_sha256": calculate_source_artifact_sha256(checkpoint),
        "plan_sha256": initial.plan.plan_sha256,
        "config_sha256": initial.plan.config.config_sha256,
        "dataset_sha256": initial.plan.config.dataset_sha256,
        "source_artifact_sha256": initial.plan.config.source_artifact_sha256,
        "execution_config_sha256": initial.execution_config_sha256,
        "execution_plan_sha256": initial.execution_plan_sha256,
    }
    metadata_path = checkpoint_root / "checkpoint.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    resumed = build_training_run(
        config_path=CONFIG_PATH,
        environment=environment,
        resume_metadata_path=metadata_path,
        dry_run=True,
    )

    assert [recipe.kind.value for recipe in pending_stage_recipes(resumed)] == [
        "preference_distillation"
    ]


def test_maintainer_scripts_support_dependency_free_dry_runs(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    write_source(raw)
    planned = tmp_path / "planned"
    prepare_process = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_dataset.py",
            "--input",
            str(raw),
            "--output-dir",
            str(planned),
            "--expected-input-sha256",
            sha256(raw),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert prepare_process.returncode == 0, prepare_process.stderr
    assert json.loads(prepare_process.stdout)["product_id"] == "metaflora-incubus-v1"
    assert not planned.exists()

    manifest, source, _ = prepare_inputs(tmp_path)
    environment = {**os.environ, **run_environment(CONFIG_PATH, manifest, source)}
    train_process = subprocess.run(
        [
            sys.executable,
            "scripts/train_candidate.py",
            "--config",
            str(CONFIG_PATH),
            "--dry-run",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert train_process.returncode == 0, train_process.stderr
    output = json.loads(train_process.stdout)
    assert output["product_id"] == "metaflora-incubus-v1"
    assert output["stages"] == ["sft", "preference_distillation"]
    assert all(
        name not in (prepare_process.stdout + train_process.stdout).casefold()
        for name in ("forbidden-build-input", "forbidden-teacher-a", "forbidden-teacher-b")
    )
