from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from metaflora_incubus.gguf_benchmark_runner import (
    BenchmarkRunnerConfig,
    GgufBenchmarkError,
    load_benchmark_cases,
    run_gguf_benchmark,
    score_response,
)

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "benchmarks" / "gguf-v1-cases.jsonl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeProcess:
    def __init__(self) -> None:
        self.terminated = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.terminated = True


def config(tmp_path: Path) -> BenchmarkRunnerConfig:
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"pinned-server")
    binary.chmod(0o700)
    model = tmp_path / "candidate.gguf"
    model.write_bytes(b"GGUFpinned-model")
    return BenchmarkRunnerConfig.create(
        server_binary=binary,
        server_sha256=digest(binary),
        model_path=model,
        model_sha256=digest(model),
        cases_path=CASES_PATH,
        output_dir=tmp_path / "evidence",
        seed=4242,
        port=18081,
        health_timeout_seconds=2.0,
        request_timeout_seconds=5.0,
    )


def test_committed_case_bank_has_six_dimensions_and_eight_safe_cases_each() -> None:
    cases = load_benchmark_cases(CASES_PATH)
    counts: dict[str, int] = {}
    languages: dict[str, set[str]] = {}
    serialized = CASES_PATH.read_text(encoding="utf-8").lower()

    for case in cases:
        counts[case.dimension] = counts.get(case.dimension, 0) + 1
        languages.setdefault(case.dimension, set()).add(case.language)

    assert len(cases) >= 48
    assert counts == {
        "coding": 8,
        "tool_calling": 8,
        "agentic_search": 8,
        "text_quality": 8,
        "russian": 8,
        "english": 8,
    }
    assert all(value == {"ru", "en"} for value in languages.values())
    assert "source_model" not in serialized
    assert all(term not in serialized for term in ("malware", "exploit", "credential theft"))


def test_runner_starts_pinned_server_and_writes_bound_evidence(tmp_path: Path) -> None:
    active_config = config(tmp_path)
    process = FakeProcess()
    commands: list[tuple[list[str], dict[str, str]]] = []
    posts: list[dict[str, Any]] = []

    def process_factory(command: list[str], environment: dict[str, str]) -> FakeProcess:
        commands.append((command, environment))
        return process

    def http_client(
        method: str, url: str, payload: dict[str, Any] | None, timeout: float
    ) -> tuple[int, dict[str, Any]]:
        if method == "GET":
            return 200, {"status": "ok"}
        assert payload is not None
        posts.append(payload)
        return 200, {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Clear concise response with requested details."},
                }
            ]
        }

    ticks = iter(float(value) for value in range(200))
    evidence = run_gguf_benchmark(
        active_config,
        process_factory=process_factory,
        http_client=http_client,
        monotonic=lambda: next(ticks),
        sleeper=lambda _: None,
    )

    assert len(commands) == 1
    command, environment = commands[0]
    assert command[0] == str(active_config.server_binary)
    assert command[command.index("--model") + 1] == str(active_config.model_path)
    assert command[command.index("--seed") + 1] == "4242"
    assert command[command.index("--gpu-layers") + 1] == "999"
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert "HF_TOKEN" not in environment
    assert len(posts) == 48
    assert all(
        post["seed"] == 4242 and post["temperature"] == 0 and post["max_tokens"] == 512
        for post in posts
    )
    assert process.terminated is True
    assert evidence["artifact_sha256"] == active_config.model_sha256
    assert set(evidence["scores"]) == {
        "coding",
        "tool_calling",
        "agentic_search",
        "text_quality",
        "russian",
        "english",
    }

    cases_rows = [
        json.loads(line)
        for line in (active_config.output_dir / "benchmark-cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    raw_rows = [
        json.loads(line)
        for line in (active_config.output_dir / "benchmark-raw.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    stored_evidence = json.loads(
        (active_config.output_dir / "benchmark-evidence.json").read_text(encoding="utf-8")
    )
    assert len(cases_rows) == len(raw_rows) == 48
    assert all(row["artifact_sha256"] == active_config.model_sha256 for row in raw_rows)
    assert all(row["seed"] == 4242 and row["latency_ms"] == 1000.0 for row in raw_rows)
    assert all(
        "raw_response" in row and "response" in row and "scores" in row and "refused" in row
        for row in raw_rows
    )
    assert all("tool_call_parse" in row for row in raw_rows)
    assert stored_evidence == evidence
    assert stored_evidence["raw_sha256"] == digest(active_config.output_dir / "benchmark-raw.jsonl")


def test_runner_fails_before_process_on_artifact_mismatch(tmp_path: Path) -> None:
    active_config = config(tmp_path)
    active_config.model_path.write_bytes(b"GGUFtampered")
    called = False

    def process_factory(command: list[str], environment: dict[str, str]) -> FakeProcess:
        nonlocal called
        called = True
        return FakeProcess()

    with pytest.raises(GgufBenchmarkError, match="model SHA-256"):
        run_gguf_benchmark(active_config, process_factory=process_factory)
    assert called is False
    assert not active_config.output_dir.exists()


def test_runner_fails_closed_on_malformed_chat_response(tmp_path: Path) -> None:
    active_config = config(tmp_path)
    process = FakeProcess()

    def http_client(
        method: str, url: str, payload: dict[str, Any] | None, timeout: float
    ) -> tuple[int, dict[str, Any]]:
        if method == "GET":
            return 200, {"status": "ok"}
        return 200, {"choices": []}

    with pytest.raises(GgufBenchmarkError, match="chat response"):
        run_gguf_benchmark(
            active_config,
            process_factory=lambda command, environment: process,
            http_client=http_client,
            sleeper=lambda _: None,
        )
    assert process.terminated is True
    assert not (active_config.output_dir / "benchmark-evidence.json").exists()


def test_rule_scoring_is_deterministic_and_parses_tool_call() -> None:
    tool_case = next(
        case for case in load_benchmark_cases(CASES_PATH) if case.dimension == "tool_calling"
    )
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool_case.expected_tool_name,
                                "arguments": json.dumps(dict(tool_case.expected_tool_arguments)),
                            },
                        }
                    ],
                },
            }
        ]
    }

    first = score_response(tool_case, response)
    second = score_response(tool_case, response)

    assert first == second
    assert first.score == 1.0
    assert first.refused is False
    assert first.tool_call_parse["valid"] is True
