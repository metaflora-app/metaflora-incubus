# Metaflora Incubus v1

Metaflora Incubus v1 is a compact local model for programming, agentic work,
search, and Russian and English text. The release target is a 5–6 GB download
that runs through a loopback API and appears in OpenCode without manual provider
configuration.

This repository contains the model-building, evaluation, release-gate, and
installer code. It does not contain a finished checkpoint yet. A public model
link and one-command installer will be added only after the quantized artifact
passes the complete release suite.

## Release requirements

The final Q4 artifact must:

- beat the frozen internal reference and the declared local competitors on
  one reproducible benchmark harness;
- win the required coding, tool-use, agentic-search, text, Russian, and English
  groups without a critical regression;
- reduce unjustified refusals on a held-out set while preserving answer quality;
- stay within the 6 GB download limit;
- pass a clean install, transactional failed-update recovery, OpenCode discovery, and uninstall on
  every supported operating system.

If the optional speech package ships with v1, it receives its own download and
must pass the published WER/CER gate. Speech files never increase the mandatory
text-model download.

The durable product and engineering contract lives in
[`docs/product/incubus-v1-capability.md`](docs/product/incubus-v1-capability.md).

## Current developer commands

Create a Python 3.11 environment and install the repository:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The maintainer-only research builder accepts a locally configured build input:

```sh
export INCUBUS_BUILD_INPUT="$HOME/Models/incubus-build-input"

incubus doctor --model-size-gb 18 --required-vram-gb 24
incubus targets --model "$INCUBUS_BUILD_INPUT"
incubus run \
  --model "$INCUBUS_BUILD_INPUT" \
  --calibration examples/calibration.jsonl \
  --output runs/incubus-v1-candidate \
  --strength 0.85
```

These commands build a research candidate. They do not create a releasable v1
by themselves. Training, distillation, quantization, competitor evaluation, and
the release gate still have to pass.

After the training stages finish, the maintainer export command merges the
adapter into local safetensors, converts the merged checkpoint to GGUF and
creates the Q4 release candidate. It deliberately leaves `release_ready=false`
until parity tests and the pinned release gates pass:

```sh
python scripts/export_candidate.py \
  --base "$INCUBUS_BUILD_INPUT" \
  --adapter artifacts/incubus-v1/preference_distillation/final \
  --convert-script "$INCUBUS_GGUF_CONVERTER" \
  --quantize-binary "$INCUBUS_GGUF_QUANTIZER" \
  --output artifacts/incubus-v1/export
```

## Available commands

| Command | Purpose |
| --- | --- |
| `incubus doctor` | Check disk, RAM, and accelerator resources. |
| `incubus targets` | Inspect compatible projection matrices. |
| `incubus run` | Build a separate research candidate. |
| `incubus evaluate` | Evaluate answers with a local semantic judge. |
| `incubus serve` | Run a candidate on `127.0.0.1`. |

## Security and data boundaries

- Model and tokenizer loading disables remote code execution.
- The inference server listens only on `127.0.0.1`.
- Evaluation data stays local unless the operator explicitly invokes a network
  search tool.
- Release files require a signed manifest and SHA-256 verification.
- Installation state is immutable, updates are staged, and uninstall removes
  only files owned by Incubus.
- Required third-party licences and notices remain in the legal release files.

## Verification

```sh
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pip check
git diff --check
```

The repository currently enforces at least 80% test coverage. Benchmark scores
are never inferred from unit tests; real model outputs and raw reports are
required before publication.
