"""Deterministic local GGUF benchmark execution with artifact-bound evidence."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

DIMENSIONS = (
    "coding",
    "tool_calling",
    "agentic_search",
    "text_quality",
    "russian",
    "english",
)
LANGUAGES = ("ru", "en")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")
_LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}")
PRODUCTION_ATTESTATION_PUBLIC_KEY = "eqUEQBjrmtGSwGRtxYBiui3L7s0MzV_mx28PFLjTUA8="
PINNED_V1_CASE_BANK_SHA256 = "9f18aba6bed35a1165cb5015ab10302f6a219c10ea1e564838a77cc3bcd75d49"
_REFUSAL_MARKERS = (
    "i can't help",
    "i cannot help",
    "i'm unable to",
    "cannot comply",
    "не могу помочь",
    "не могу выполнить",
    "не буду отвечать",
)


class GgufBenchmarkError(RuntimeError):
    """Raised when benchmark evidence cannot be completed safely."""


class ServerProcess(Protocol):
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


HttpClient = Callable[[str, str, dict[str, Any] | None, float], tuple[int, dict[str, Any]]]
ProcessFactory = Callable[[list[str], dict[str, str]], ServerProcess]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    dimension: str
    language: str
    prompt: str
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...] = ()
    expected_tool_name: str | None = None
    expected_tool_arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_terms", tuple(self.required_terms))
        object.__setattr__(self, "forbidden_terms", tuple(self.forbidden_terms))
        object.__setattr__(
            self,
            "expected_tool_arguments",
            MappingProxyType(dict(self.expected_tool_arguments)),
        )


@dataclass(frozen=True)
class ScoredResponse:
    score: float
    refused: bool
    content: str
    finish_reason: str
    tool_call_parse: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_call_parse", MappingProxyType(dict(self.tool_call_parse)))


@dataclass(frozen=True)
class BenchmarkRunnerConfig:
    server_binary: Path
    server_sha256: str
    model_path: Path
    model_sha256: str
    cases_path: Path
    output_dir: Path
    seed: int
    port: int
    health_timeout_seconds: float
    request_timeout_seconds: float
    runner_code_revision: str
    gpu_layers: int

    @classmethod
    def create(cls, **values: object) -> BenchmarkRunnerConfig:
        server_sha = _validated_sha(values.get("server_sha256"), "server_sha256")
        model_sha = _validated_sha(values.get("model_sha256"), "model_sha256")
        seed = _positive_int(values.get("seed"), "seed", allow_zero=True)
        port = _positive_int(values.get("port"), "port")
        if port > 65535:
            raise GgufBenchmarkError("port is outside valid range")
        health_timeout = _positive_float(
            values.get("health_timeout_seconds"), "health_timeout_seconds"
        )
        request_timeout = _positive_float(
            values.get("request_timeout_seconds"), "request_timeout_seconds"
        )
        gpu_layers = _positive_int(values.get("gpu_layers"), "gpu_layers", allow_zero=True)
        runner_revision = values.get("runner_code_revision")
        if not isinstance(runner_revision, str) or _LOWER_HEX_40.fullmatch(runner_revision) is None:
            raise GgufBenchmarkError("runner_code_revision must be exact lowercase revision")
        return cls(
            server_binary=_required_path(values.get("server_binary"), "server_binary"),
            server_sha256=server_sha,
            model_path=_required_path(values.get("model_path"), "model_path"),
            model_sha256=model_sha,
            cases_path=_required_path(values.get("cases_path"), "cases_path"),
            output_dir=_required_path(values.get("output_dir"), "output_dir"),
            seed=seed,
            port=port,
            health_timeout_seconds=health_timeout,
            request_timeout_seconds=request_timeout,
            runner_code_revision=runner_revision,
            gpu_layers=gpu_layers,
        )


def load_benchmark_cases(path: Path) -> tuple[BenchmarkCase, ...]:
    """Load and validate the committed case bank, failing on incomplete coverage."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GgufBenchmarkError("benchmark case bank cannot be read") from exc
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise GgufBenchmarkError(f"empty benchmark case line: {line_number}")
        try:
            document = json.loads(line)
            case = _parse_case(document)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GgufBenchmarkError(f"invalid benchmark case line: {line_number}") from exc
        cases.append(case)
    identifiers = tuple(case.case_id for case in cases)
    if len(set(identifiers)) != len(identifiers):
        raise GgufBenchmarkError("benchmark case IDs must be unique")
    counts = Counter(case.dimension for case in cases)
    if set(counts) != set(DIMENSIONS) or any(counts[item] < 8 for item in DIMENSIONS):
        raise GgufBenchmarkError("benchmark case bank lacks required dimension coverage")
    for dimension in DIMENSIONS:
        languages = {case.language for case in cases if case.dimension == dimension}
        if languages != set(LANGUAGES):
            raise GgufBenchmarkError(f"benchmark dimension lacks language coverage: {dimension}")
    return tuple(cases)


def score_response(case: BenchmarkCase, response: Mapping[str, object]) -> ScoredResponse:
    """Apply deterministic, local rules to a validated chat response."""
    message, finish_reason = _extract_message(response)
    content_value = message.get("content", "")
    if content_value is None:
        content_value = ""
    if not isinstance(content_value, str):
        raise GgufBenchmarkError("chat response content must be text")
    normalized = content_value.casefold()
    refused = any(marker in normalized for marker in _REFUSAL_MARKERS)
    tool_parse = _parse_tool_call(message)
    if case.expected_tool_name is not None:
        name_score = 0.6 if tool_parse.get("name") == case.expected_tool_name else 0.0
        arguments_score = (
            0.4 if tool_parse.get("arguments") == dict(case.expected_tool_arguments) else 0.0
        )
        score = name_score + arguments_score
    elif not case.required_terms:
        score = 1.0 if normalized.strip() else 0.0
    else:
        hits = sum(term.casefold() in normalized for term in case.required_terms)
        score = hits / len(case.required_terms)
    if refused or any(term.casefold() in normalized for term in case.forbidden_terms):
        score = 0.0
    return ScoredResponse(
        score=round(score, 6),
        refused=refused,
        content=content_value,
        finish_reason=finish_reason,
        tool_call_parse=tool_parse,
    )


def run_gguf_benchmark(
    config: BenchmarkRunnerConfig,
    *,
    process_factory: ProcessFactory | None = None,
    http_client: HttpClient | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    _attestation_signer: Callable[[bytes], bytes] | None = None,
) -> dict[str, object]:
    """Run every case against a pinned local server and atomically write evidence."""
    _verify_runtime_artifact(config.server_binary, config.server_sha256, "server binary")
    _verify_runtime_artifact(
        config.model_path, config.model_sha256, "model", required_magic=b"GGUF"
    )
    cases = load_benchmark_cases(config.cases_path)
    factory = process_factory or _default_process_factory
    client = http_client or _default_http_client
    base_url = f"http://127.0.0.1:{config.port}"
    process = factory(_server_command(config), _minimal_environment())
    raw_rows: list[dict[str, object]] = []
    try:
        _wait_for_health(
            process,
            client,
            base_url,
            config.health_timeout_seconds,
            monotonic,
            sleeper,
        )
        for case in cases:
            payload = _request_payload(case, config.seed)
            started = monotonic()
            status, response = client(
                "POST",
                f"{base_url}/v1/chat/completions",
                payload,
                config.request_timeout_seconds,
            )
            latency_ms = round((monotonic() - started) * 1000, 3)
            if status != 200:
                raise GgufBenchmarkError(f"chat request failed for {case.case_id}")
            scored = score_response(case, response)
            raw_rows.append(
                {
                    "artifact_sha256": config.model_sha256,
                    "case_id": case.case_id,
                    "dimension": case.dimension,
                    "finish_reason": scored.finish_reason,
                    "language": case.language,
                    "latency_ms": latency_ms,
                    "raw_response": response,
                    "refused": scored.refused,
                    "response": scored.content,
                    "score": scored.score,
                    "scores": {case.dimension: scored.score},
                    "seed": config.seed,
                    "tool_call_parse": dict(scored.tool_call_parse),
                }
            )
    finally:
        _stop_process(process)
    if len(raw_rows) != len(cases):
        raise GgufBenchmarkError("benchmark run is incomplete")
    return _write_evidence(
        config,
        cases,
        raw_rows,
        attestation_signer=_attestation_signer or _production_attestation_signer,
    )


def _parse_case(document: object) -> BenchmarkCase:
    if not isinstance(document, dict):
        raise TypeError("case must be an object")
    case_id = _nonempty_string(document["case_id"], "case_id")
    dimension = _nonempty_string(document["dimension"], "dimension")
    language = _nonempty_string(document["language"], "language")
    prompt = _nonempty_string(document["prompt"], "prompt")
    if dimension not in DIMENSIONS or language not in LANGUAGES:
        raise ValueError("unsupported benchmark dimension or language")
    required = _string_tuple(document.get("required_terms", []), "required_terms")
    forbidden = _string_tuple(document.get("forbidden_terms", []), "forbidden_terms")
    tool_name_value = document.get("expected_tool_name")
    tool_name = None if tool_name_value is None else _nonempty_string(tool_name_value, "tool")
    arguments = document.get("expected_tool_arguments", {})
    if not isinstance(arguments, dict):
        raise TypeError("expected_tool_arguments must be an object")
    if dimension in {"tool_calling", "agentic_search"} and tool_name is None:
        raise ValueError("tool dimensions require an expected tool")
    return BenchmarkCase(
        case_id=case_id,
        dimension=dimension,
        language=language,
        prompt=prompt,
        required_terms=required,
        forbidden_terms=forbidden,
        expected_tool_name=tool_name,
        expected_tool_arguments=arguments,
    )


def _extract_message(response: Mapping[str, object]) -> tuple[dict[str, object], str]:
    try:
        choices = response["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise TypeError
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise TypeError
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise TypeError
        return choice["message"], finish_reason
    except (KeyError, TypeError) as exc:
        raise GgufBenchmarkError("malformed chat response") from exc


def _parse_tool_call(message: Mapping[str, object]) -> dict[str, object]:
    calls = message.get("tool_calls")
    if calls is None:
        return {"arguments": None, "error": None, "name": None, "valid": False}
    try:
        if not isinstance(calls, list) or len(calls) != 1:
            raise TypeError("expected exactly one tool call")
        function = calls[0]["function"]
        name = function["name"]
        arguments = json.loads(function["arguments"])
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise TypeError("invalid tool call fields")
        return {"arguments": arguments, "error": None, "name": name, "valid": True}
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"arguments": None, "error": str(exc), "name": None, "valid": False}


def _request_payload(case: BenchmarkCase, seed: int) -> dict[str, object]:
    payload: dict[str, object] = {
        "max_tokens": 512,
        "messages": [{"content": case.prompt, "role": "user"}],
        "seed": seed,
        "stream": False,
        "temperature": 0,
    }
    if case.expected_tool_name is not None:
        properties = {
            key: {"type": _json_schema_type(value)}
            for key, value in case.expected_tool_arguments.items()
        }
        payload["tools"] = [
            {
                "function": {
                    "description": "Return the requested structured result.",
                    "name": case.expected_tool_name,
                    "parameters": {
                        "additionalProperties": False,
                        "properties": properties,
                        "required": sorted(properties),
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ]
    return payload


def _server_command(config: BenchmarkRunnerConfig) -> list[str]:
    return [
        str(config.server_binary),
        "--model",
        str(config.model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--seed",
        str(config.seed),
        "--temp",
        "0",
        "--parallel",
        "1",
        "--gpu-layers",
        str(config.gpu_layers),
        "--ctx-size",
        "4096",
    ]


def _wait_for_health(
    process: ServerProcess,
    client: HttpClient,
    base_url: str,
    timeout: float,
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    deadline = monotonic() + timeout
    while monotonic() <= deadline:
        if process.poll() is not None:
            raise GgufBenchmarkError(
                "llama-server exited before health check" + _server_stderr_diagnostic(process)
            )
        try:
            status, response = client("GET", f"{base_url}/health", None, 1.0)
            if status == 200 and response.get("status") in {"ok", "ready"}:
                return
        except Exception:
            pass
        sleeper(0.1)
    raise GgufBenchmarkError("llama-server health check timed out")


def _write_evidence(
    config: BenchmarkRunnerConfig,
    cases: tuple[BenchmarkCase, ...],
    raw_rows: list[dict[str, object]],
    *,
    attestation_signer: Callable[[bytes], bytes],
) -> dict[str, object]:
    cases_payload = config.cases_path.read_bytes()
    raw_payload = _jsonl_bytes(raw_rows)
    cases_sha = hashlib.sha256(cases_payload).hexdigest()
    if cases_sha != PINNED_V1_CASE_BANK_SHA256:
        raise GgufBenchmarkError("benchmark case bank is not the pinned release suite")
    raw_sha = hashlib.sha256(raw_payload).hexdigest()
    scores = {
        dimension: round(
            sum(float(row["score"]) for row in raw_rows if row["dimension"] == dimension)
            / sum(row["dimension"] == dimension for row in raw_rows),
            6,
        )
        for dimension in DIMENSIONS
    }
    evidence: dict[str, object] = {
        "artifact_sha256": config.model_sha256,
        "case_count": len(cases),
        "cases_sha256": cases_sha,
        "raw_sha256": raw_sha,
        "refusal_rate": round(sum(bool(row["refused"]) for row in raw_rows) / len(raw_rows), 6),
        "runtime_sha256": config.server_sha256,
        "scores": scores,
        "seed": config.seed,
        "settings": {
            "gpu_layers": config.gpu_layers,
            "max_tokens": 512,
            "parallel": 1,
            "stream": False,
            "temperature": 0,
        },
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(config.output_dir / "benchmark-cases.jsonl", cases_payload)
    _atomic_write(config.output_dir / "benchmark-raw.jsonl", raw_payload)
    attestation = {
        "artifact_sha256": config.model_sha256,
        "dataset_sha256": cases_sha,
        "raw_output_sha256": raw_sha,
        "runner_code_revision": config.runner_code_revision,
        "sample_count": len(raw_rows),
        "schema_version": 1,
        "seeds": [config.seed],
    }
    attestation_payload = _canonical_json(attestation) + b"\n"
    signature = attestation_signer(attestation_payload)
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise GgufBenchmarkError("benchmark attestation signer returned an invalid signature")
    _atomic_write(config.output_dir / "benchmark-attestation.json", attestation_payload)
    _atomic_write(config.output_dir / "benchmark-attestation.sig", signature)
    _atomic_write(
        config.output_dir / "benchmark-evidence.json",
        _canonical_json(evidence) + b"\n",
    )
    return evidence


def _production_attestation_signer(payload: bytes) -> bytes:
    encoded = os.environ.get("INCUBUS_BENCHMARK_SIGNING_KEY", "")
    try:
        private_bytes = base64.urlsafe_b64decode(encoded.encode("ascii"))
        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        public_bytes = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        expected_public = base64.urlsafe_b64decode(
            PRODUCTION_ATTESTATION_PUBLIC_KEY.encode("ascii")
        )
    except (ValueError, binascii.Error, UnicodeEncodeError) as exc:
        raise GgufBenchmarkError("benchmark signing key is invalid") from exc
    if public_bytes != expected_public:
        raise GgufBenchmarkError("benchmark signing key does not match the production key")
    return private_key.sign(payload)


def _case_document(case: BenchmarkCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "dimension": case.dimension,
        "expected_tool_arguments": dict(case.expected_tool_arguments),
        "expected_tool_name": case.expected_tool_name,
        "forbidden_terms": case.forbidden_terms,
        "language": case.language,
        "prompt": case.prompt,
        "required_terms": case.required_terms,
    }


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _verify_runtime_artifact(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    required_magic: bytes | None = None,
) -> None:
    if not path.is_file():
        raise GgufBenchmarkError(f"{label} is missing")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        magic = stream.read(len(required_magic or b""))
        digest.update(magic)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if required_magic is not None and magic != required_magic:
        raise GgufBenchmarkError(f"{label} is not a GGUF artifact")
    if digest.hexdigest() != expected_sha256:
        raise GgufBenchmarkError(f"{label} SHA-256 mismatch")


def _minimal_environment() -> dict[str, str]:
    allowed = ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "TMPDIR")
    return {key: os.environ[key] for key in allowed if key in os.environ}


def _server_stderr_diagnostic(process: ServerProcess) -> str:
    """Expose a bounded startup diagnostic without retaining normal benchmark output."""
    stream = getattr(process, "stderr", None)
    if stream is None:
        return ""
    try:
        payload = stream.read()
    except (AttributeError, OSError, ValueError):
        return ""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    if not isinstance(payload, str):
        return ""
    compact = " ".join(payload.strip().split())[-2000:]
    return f": {compact}" if compact else ""


def _default_process_factory(command: list[str], environment: dict[str, str]) -> ServerProcess:
    return subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _default_http_client(
    method: str, url: str, payload: dict[str, Any] | None, timeout: float
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else _canonical_json(payload)
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            document = json.loads(response.read())
            if not isinstance(document, dict):
                raise GgufBenchmarkError("HTTP response is not an object")
            return response.status, document
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise GgufBenchmarkError("local HTTP request failed") from exc


def _stop_process(process: ServerProcess) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _validated_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise GgufBenchmarkError(f"{name} must be exact lowercase SHA-256")
    return value


def _required_path(value: object, name: str) -> Path:
    if not isinstance(value, Path):
        raise GgufBenchmarkError(f"{name} must be a Path")
    return value


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < lower:
        raise GgufBenchmarkError(f"{name} must be an integer >= {lower}")
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise GgufBenchmarkError(f"{name} must be positive")
    return float(value)


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value.strip()


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise TypeError(f"{name} must be a string list")
    return tuple(item.strip() for item in value)


def _json_schema_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"
