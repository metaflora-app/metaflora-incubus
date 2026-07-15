from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from metaflora_incubus.head_to_head_benchmark import (
    ComparisonModel,
    HeadToHeadConfig,
    HeadToHeadError,
    HeadToHeadPolicy,
    compare_runs,
    load_head_to_head_config,
    run_head_to_head,
)

DIMENSION_CASES = {
    "coding": ("coding-en-01", "coding-ru-02"),
    "tool_calling": ("tool-en-01", "tool-ru-02"),
    "agentic_search": ("search-en-01", "search-ru-02"),
    "text_quality": ("text-en-01", "text-ru-02"),
    "russian": ("russian-ru-01", "russian-en-02"),
    "english": ("english-en-01", "english-ru-02"),
}


def _rows(score: float, *, refused: bool = False) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for dimension, case_ids in DIMENSION_CASES.items():
        for index, case_id in enumerate(case_ids):
            rows.append(
                {
                    "artifact_sha256": "a" * 64,
                    "case_id": case_id,
                    "dimension": dimension,
                    "language": "en" if index == 0 else "ru",
                    "refused": refused,
                    "score": score,
                    "seed": 4242,
                }
            )
    return tuple(rows)


def _config(tmp_path: Path, models: tuple[ComparisonModel, ...]) -> HeadToHeadConfig:
    server = tmp_path / "llama-server"
    server.write_bytes(b"server")
    cases = tmp_path / "cases.jsonl"
    cases.write_text("{}\n", encoding="utf-8")
    return HeadToHeadConfig.create(
        server_binary=server,
        server_sha256=hashlib.sha256(server.read_bytes()).hexdigest(),
        cases_path=cases,
        output_dir=tmp_path / "head-to-head",
        runner_code_revision="1" * 40,
        models=models,
        seed=4242,
        port=18100,
        gpu_layers=999,
        health_timeout_seconds=120.0,
        request_timeout_seconds=120.0,
    )


def _model(tmp_path: Path, model_id: str, role: str) -> ComparisonModel:
    path = tmp_path / f"{model_id}.gguf"
    path.write_bytes(b"GGUF" + model_id.encode())
    return ComparisonModel.create(
        model_id=model_id,
        role=role,
        model_path=path,
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_statistical_promotion_requires_significant_paired_lead() -> None:
    runs = {
        "candidate-v1": _rows(1.0),
        "incumbent": _rows(0.0),
        "same-size-a": _rows(0.0),
    }
    roles = {
        "candidate-v1": "candidate",
        "incumbent": "incumbent",
        "same-size-a": "same_size",
    }

    report = compare_runs(runs, roles, HeadToHeadPolicy.release())

    assert report["verdict"] == "promotion_passed"
    assert report["public_winner_claim"] is False
    assert report["required_comparators"] == ["incumbent", "same-size-a"]
    assert all(
        comparison["overall_delta_ci95"][0] > 0 for comparison in report["comparisons"].values()
    )


def test_tie_is_not_statistical_victory() -> None:
    runs = {
        "candidate-v1": _rows(0.75),
        "incumbent": _rows(0.75),
        "same-size-a": _rows(0.75),
    }
    roles = {
        "candidate-v1": "candidate",
        "incumbent": "incumbent",
        "same-size-a": "same_size",
    }

    report = compare_runs(runs, roles, HeadToHeadPolicy.release())

    assert report["verdict"] == "not_demonstrated"
    assert "overall_lead_not_significant:incumbent" in report["failures"]


def test_capability_regression_blocks_promotion_despite_overall_lead() -> None:
    candidate = list(_rows(1.0))
    for row in candidate:
        if row["dimension"] == "coding":
            row["score"] = 0.0
    runs = {
        "candidate-v1": tuple(candidate),
        "incumbent": _rows(0.1),
        "same-size-a": _rows(0.1),
    }
    roles = {
        "candidate-v1": "candidate",
        "incumbent": "incumbent",
        "same-size-a": "same_size",
    }

    report = compare_runs(runs, roles, HeadToHeadPolicy.release())

    assert report["verdict"] == "not_demonstrated"
    assert "capability_regression:incumbent:coding" in report["failures"]


def test_benign_overrefusal_increase_blocks_promotion() -> None:
    candidate = list(_rows(1.0))
    candidate[0]["refused"] = True
    runs = {
        "candidate-v1": tuple(candidate),
        "incumbent": _rows(0.0),
        "same-size-a": _rows(0.0),
    }
    roles = {
        "candidate-v1": "candidate",
        "incumbent": "incumbent",
        "same-size-a": "same_size",
    }

    report = compare_runs(runs, roles, HeadToHeadPolicy.release())

    assert report["verdict"] == "not_demonstrated"
    assert "overrefusal_regression:incumbent" in report["failures"]


def test_comparison_rejects_nonidentical_case_sets() -> None:
    candidate = _rows(1.0)
    baseline = _rows(0.0)[:-1]

    with pytest.raises(HeadToHeadError, match="case sets"):
        compare_runs(
            {"candidate-v1": candidate, "incumbent": baseline, "same-size-a": baseline},
            {
                "candidate-v1": "candidate",
                "incumbent": "incumbent",
                "same-size-a": "same_size",
            },
            HeadToHeadPolicy.release(),
        )


def test_config_requires_candidate_incumbent_and_same_size(tmp_path: Path) -> None:
    with pytest.raises(HeadToHeadError, match="roles"):
        _config(tmp_path, (_model(tmp_path, "candidate-v1", "candidate"),))


def test_conditional_bonsai_failure_is_recorded_without_aborting(tmp_path: Path) -> None:
    models = (
        _model(tmp_path, "candidate-v1", "candidate"),
        _model(tmp_path, "incumbent", "incumbent"),
        _model(tmp_path, "same-size-a", "same_size"),
        _model(tmp_path, "bonsai", "conditional"),
    )
    active_config = _config(tmp_path, models)

    def benchmark_runner(config: object) -> dict[str, object]:
        model_id = Path(config.model_path).stem  # type: ignore[attr-defined]
        if model_id == "bonsai":
            raise RuntimeError("unsupported tensor type")
        output_dir = config.output_dir  # type: ignore[attr-defined]
        output_dir.mkdir(parents=True)
        score = 1.0 if model_id == "candidate-v1" else 0.0
        with (output_dir / "benchmark-raw.jsonl").open("w", encoding="utf-8") as stream:
            for row in _rows(score):
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        return {"scores": {"coding": score}}

    report = run_head_to_head(active_config, benchmark_runner=benchmark_runner)

    assert report["verdict"] == "promotion_passed"
    assert report["conditional_models"] == {
        "bonsai": {"status": "unavailable", "reason": "benchmark_failed"}
    }
    stored = json.loads(
        (active_config.output_dir / "head-to-head-report.json").read_text(encoding="utf-8")
    )
    assert stored == report


def test_mandatory_model_failure_fails_closed(tmp_path: Path) -> None:
    models = (
        _model(tmp_path, "candidate-v1", "candidate"),
        _model(tmp_path, "incumbent", "incumbent"),
        _model(tmp_path, "same-size-a", "same_size"),
    )
    active_config = _config(tmp_path, models)

    with pytest.raises(HeadToHeadError, match="mandatory benchmark failed"):
        run_head_to_head(
            active_config,
            benchmark_runner=lambda config: (_ for _ in ()).throw(RuntimeError("boom")),
        )


def test_manifest_loads_relative_paths_and_exact_roles(tmp_path: Path) -> None:
    server = tmp_path / "llama-server"
    server.write_bytes(b"server")
    cases = tmp_path / "cases.jsonl"
    cases.write_text("{}\n", encoding="utf-8")
    model_documents = []
    for model_id, role in (
        ("candidate-v1", "candidate"),
        ("incumbent", "incumbent"),
        ("same-size-a", "same_size"),
        ("bonsai", "conditional"),
    ):
        model = tmp_path / f"{model_id}.gguf"
        model.write_bytes(b"GGUF" + model_id.encode())
        model_documents.append(
            {
                "id": model_id,
                "role": role,
                "path": model.name,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "head-to-head.json"
    manifest.write_text(
        json.dumps(
            {
                "server_binary": server.name,
                "server_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
                "cases_path": cases.name,
                "output_dir": "comparison-output",
                "runner_code_revision": "1" * 40,
                "seed": 4242,
                "port": 18100,
                "gpu_layers": 999,
                "health_timeout_seconds": 120,
                "request_timeout_seconds": 120,
                "models": model_documents,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_head_to_head_config(manifest)

    assert loaded.server_binary == server
    assert loaded.output_dir == tmp_path / "comparison-output"
    assert [model.role for model in loaded.models] == [
        "candidate",
        "incumbent",
        "same_size",
        "conditional",
    ]
