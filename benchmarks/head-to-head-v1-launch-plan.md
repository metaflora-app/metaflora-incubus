# Private GGUF head-to-head v1

This run compares one candidate with the incumbent and two required same-size Q5
baselines. Bonsai is conditional: a load or inference failure is recorded without
invalidating the required matrix. The report is private evidence for a promotion
decision. `promotion_passed` is not a public winner claim.

## 1. Fixed paths

Run from a fresh Kaggle GPU session after the candidate and incumbent GGUF files
have been restored.

```bash
export REPO=/kaggle/working/metaflora-incubus
export MODEL_DIR=/kaggle/working/head-to-head-models
export CANDIDATE_GGUF=/kaggle/working/candidate-v1.gguf
export INCUMBENT_GGUF=/kaggle/working/incumbent-v1.gguf
export LLAMA_SERVER=/kaggle/working/incubus-work/artifacts/llama-server
export HEAD_TO_HEAD_MANIFEST=/kaggle/working/head-to-head-v1.json
export HEAD_TO_HEAD_OUTPUT=/kaggle/working/head-to-head-output
mkdir -p "$MODEL_DIR" "$HEAD_TO_HEAD_OUTPUT"
cd "$REPO"
```

## 2. Download the pinned comparison artifacts

The repository revisions, file sizes, SHA-256 values, licenses and official source
model cards are fixed in `benchmarks/head-to-head-v1-baselines.json`.

```bash
hf download bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF \
  Qwen_Qwen3-4B-Instruct-2507-Q5_K_M.gguf \
  --revision ae44f08e1392f39c0e474af10c3ff8355c8b6688 \
  --local-dir "$MODEL_DIR"

hf download bartowski/microsoft_Phi-4-mini-instruct-GGUF \
  microsoft_Phi-4-mini-instruct-Q5_K_M.gguf \
  --revision 7ff82c2aaa4dde30121698a973765f39be5288c0 \
  --local-dir "$MODEL_DIR"
```

Download Bonsai only when at least 4 GiB of additional working disk remains:

```bash
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf \
  --revision 0cf7e3d21581b169b4df1de8bf01316000e2fbb7 \
  --local-dir "$MODEL_DIR"
```

## 3. Verify and create the execution manifest

This cell checks every local file before the server starts. It also checks the
candidate, incumbent and runtime magic or executable bit. No model is selected by
an unpinned Hub branch.

```bash
sha256sum \
  "$CANDIDATE_GGUF" \
  "$INCUMBENT_GGUF" \
  "$LLAMA_SERVER" \
  "$MODEL_DIR/Qwen_Qwen3-4B-Instruct-2507-Q5_K_M.gguf" \
  "$MODEL_DIR/microsoft_Phi-4-mini-instruct-Q5_K_M.gguf"

python - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

repo = Path(os.environ["REPO"])
model_dir = Path(os.environ["MODEL_DIR"])
catalog = json.loads(
    (repo / "benchmarks/head-to-head-v1-baselines.json").read_text(encoding="utf-8")
)

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def gguf(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic != b"GGUF":
        raise RuntimeError(f"not a GGUF file: {path}")
    return digest(path), path.stat().st_size

models = []
for model_id, role, variable in (
    ("candidate-v1", "candidate", "CANDIDATE_GGUF"),
    ("incumbent-v1", "incumbent", "INCUMBENT_GGUF"),
):
    path = Path(os.environ[variable])
    sha256, _ = gguf(path)
    models.append({"id": model_id, "role": role, "path": str(path), "sha256": sha256})

for pin in catalog["baselines"]:
    path = model_dir / pin["artifact_filename"]
    if not pin["promotion_required"] and not path.exists():
        continue
    sha256, size = gguf(path)
    if sha256 != pin["artifact_sha256"] or size != pin["artifact_size_bytes"]:
        raise RuntimeError(f"baseline pin mismatch: {pin['id']}")
    models.append(
        {"id": pin["id"], "role": pin["role"], "path": str(path), "sha256": sha256}
    )

server = Path(os.environ["LLAMA_SERVER"])
if not server.is_file() or not os.access(server, os.X_OK):
    raise RuntimeError("llama-server is missing or not executable")

manifest = {
    "server_binary": str(server),
    "server_sha256": digest(server),
    "cases_path": str(repo / catalog["launch_settings"]["case_bank"]),
    "output_dir": os.environ["HEAD_TO_HEAD_OUTPUT"],
    "runner_code_revision": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip(),
    "seed": catalog["launch_settings"]["seed"],
    "port": 18100,
    "gpu_layers": catalog["launch_settings"]["gpu_layers"],
    "health_timeout_seconds": 120,
    "request_timeout_seconds": 120,
    "models": models,
}
Path(os.environ["HEAD_TO_HEAD_MANIFEST"]).write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(os.environ["HEAD_TO_HEAD_MANIFEST"])
PY
```

## 4. Run and inspect the private report

The signing key stays in the notebook secret store and must already match the
project's pinned benchmark public key.

```bash
test -n "$INCUBUS_BENCHMARK_SIGNING_KEY"
python scripts/run_head_to_head_benchmark.py \
  --manifest "$HEAD_TO_HEAD_MANIFEST"
jq '{verdict,failures,comparisons,conditional_models,public_winner_claim}' \
  "$HEAD_TO_HEAD_OUTPUT/head-to-head-report.json"
```

Promotion requires a positive lower bound of the paired 95% bootstrap interval
against the incumbent and both same-size models, no capability delta below -0.05,
and no upper confidence bound above +0.02 for benign over-refusal growth.

## GigaAM-v3 boundary

GigaAM-v3 is an ASR model with a Conformer encoder. Its input is audio and its
output is a transcription. It cannot enter this chat, code or tool-call matrix.
If speech recognition becomes a product requirement, evaluate it separately with
audio datasets and word error rate (WER).
