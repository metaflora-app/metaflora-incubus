from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "benchmarks" / "head-to-head-v1-baselines.json"
LAUNCH_PLAN = ROOT / "benchmarks" / "head-to-head-v1-launch-plan.md"


def load_catalog() -> dict[str, object]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def test_catalog_pins_two_required_open_same_size_q5_baselines() -> None:
    catalog = load_catalog()
    baselines = catalog["baselines"]
    required = [item for item in baselines if item["promotion_required"]]

    assert len(required) >= 2
    assert all(item["role"] == "same_size" for item in required)
    assert all(3_000_000_000 <= item["parameter_count"] <= 5_500_000_000 for item in required)
    assert all(item["quantization"] == "Q5_K_M" for item in required)
    assert all(item["license_id"] in {"Apache-2.0", "MIT"} for item in required)
    assert all(item["source_pipeline_tag"] == "text-generation" for item in required)


def test_every_download_is_content_and_revision_pinned() -> None:
    catalog = load_catalog()

    for item in catalog["baselines"]:
        assert re.fullmatch(r"[0-9a-f]{40}", item["source_revision"])
        assert re.fullmatch(r"[0-9a-f]{40}", item["artifact_revision"])
        assert re.fullmatch(r"[0-9a-f]{64}", item["artifact_sha256"])
        assert item["artifact_size_bytes"] > 0
        assert item["artifact_filename"].endswith(".gguf")
        assert item["source_model_card_url"].startswith("https://huggingface.co/")


def test_non_q5_or_non_same_size_models_are_advisory_only() -> None:
    baselines = load_catalog()["baselines"]
    advisory = [item for item in baselines if not item["promotion_required"]]

    assert {item["id"] for item in advisory} == {"gemma-3-4b-it", "bonsai-27b"}
    assert all(item["role"] == "conditional" for item in advisory)
    assert all(item["comparison_tier"] == "advisory" for item in advisory)


def test_gigaam_is_explicitly_excluded_from_text_llm_matrix() -> None:
    catalog = load_catalog()
    gigaam = next(item for item in catalog["excluded_models"] if item["id"] == "gigaam-v3")

    assert gigaam["source_pipeline_tag"] == "automatic-speech-recognition"
    assert gigaam["architecture_family"] == "conformer"
    assert gigaam["input_modality"] == "audio"
    assert gigaam["output_modality"] == "transcription"
    assert gigaam["include_in_text_llm_benchmark"] is False
    assert gigaam["eligible_benchmark"] == "asr_wer_only"


def test_launch_settings_match_the_committed_runner_contract() -> None:
    settings = load_catalog()["launch_settings"]

    assert settings == {
        "case_bank": "benchmarks/gguf-v1-cases.jsonl",
        "context_size": 4096,
        "gpu_layers": 999,
        "max_tokens": 512,
        "parallel": 1,
        "seed": 4242,
        "stream": False,
        "temperature": 0,
    }


def test_launch_plan_verifies_files_and_runs_private_harness() -> None:
    plan = LAUNCH_PLAN.read_text(encoding="utf-8")

    assert "head-to-head-v1-baselines.json" in plan
    assert "sha256sum" in plan
    assert "run_head_to_head_benchmark.py" in plan
    assert "INCUBUS_BENCHMARK_SIGNING_KEY" in plan
    assert "promotion_passed" in plan
    assert "public winner" in plan.lower()
    assert "GigaAM-v3" in plan and "ASR" in plan
