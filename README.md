---
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
language:
  - ru
  - en
tags:
  - gguf
  - llama.cpp
  - local
  - agentic
  - tool-use
  - vision
  - audio
---

# Metaflora Incubus v1

Metaflora Incubus v1 is a compact local model system built for code, structured
tool calls, agentic workflows, and Russian-English generation. Its routed
candidate was post-trained on **6,750 benchmark-disjoint records** spanning
bilingual instruction, rewriting, executable code, native tool dialogues,
agentic replay, and refusal reduction.

**Model download and release card:** [huggingface.co/metaflora/incubus](https://huggingface.co/metaflora/incubus)

Retained 48-case diagnostics report **100.00 agentic**, **95.83 code**,
**95.00 tool use**, **87.50 English**, **81.25 Russian**, and **81.25 text
quality**. The Q5_K_M core serves through a local OpenAI-compatible endpoint,
works offline after installation, and requires no mandatory cloud moderation
service or remote policy gateway. Voice and vision remain optional downloads,
so the primary text package stays at 3.075 GB.

## Highlights

- Compact 4B Q5_K_M deployment.
- Deterministic structured tool use and agentic search evaluation.
- Local operation without a mandatory cloud moderation service or remote
  policy gateway.
- OpenAI-compatible serving through a recent `llama-server`.
- Optional signed voice and vision packages.
- Artifact-bound benchmark receipts, checksums, and immutable revisions.

## Model overview

| Property | Value |
| --- | --- |
| Type | Causal language model |
| Parameters | 4B |
| Release format | GGUF |
| Quantization | Q5_K_M |
| File size | 3.075 GB |
| Recommended memory | 16 GiB |
| Validated context | 8,192 tokens |
| Release languages | Russian and English |

## Training and data

The routed release candidate was produced through several post-training stages
covering bilingual instruction following, text correction, executable code,
structured tool calls, agentic workflows, and refusal reduction. Training
records were selected separately from the fixed 48-case evaluation bank.

### Training scale at a glance

| Recorded quantity | Value |
| --- | ---: |
| Main continuation records | **6,750** |
| Language and writing records | **4,000** |
| Execution-oriented code records | **1,500** |
| Native tool and agentic records | **1,250** |
| Capability groups in the main mixture | **6** |
| Optimizer updates | **1,589** |
| Effective sequence slots processed | **3,178** |
| Maximum scheduled token positions | **2,440,704** |
| Tool-specialization records selected | **1,090** |
| Tool-specialization packed sequences | **992** |
| Fixed diagnostic cases | **48** |
| Cases per evaluated capability | **8** |

The 2,440,704 figure is the schedule ceiling calculated as
`1,589 updates × 2 sequences × 768 positions`. It measures the maximum number
of packed token positions presented by the schedule. The number of unique
natural-language tokens in the source corpus is a separate measurement.

### Main continuation mixture

| Data group | Records | Share | Purpose |
| --- | ---: | ---: | --- |
| Russian instruction | 1,500 | 22.22% | Russian requests, concise answers, and format control |
| English general instruction | 1,500 | 22.22% | General instruction following and answer structure |
| Rewrite and correction | 1,000 | 14.81% | Editing, grammar correction, compression, and clarity |
| Execution-oriented code | 1,500 | 22.22% | Implementation, repair, tests, and exact output behavior |
| Native structured tool calls | 750 | 11.11% | Tool selection and schema-valid arguments |
| Agentic replay | 500 | 7.41% | Multi-step planning, observation handling, and completion |
| **Total** | **6,750** | **100.00%** | |

Grouped another way, the mixture contains 3,000 direct bilingual instruction
records, 4,000 language and writing records after adding rewrite/correction,
1,500 code records, and 1,250 native tool plus agentic records. Language data
therefore accounts for 59.26% of the main continuation; specialist code,
tools, and agentic behavior account for the remaining 40.74%.

The complete mixture is content-addressed by SHA-256:
`9141185743bd8681493caac268cfb0339e95b0fd1788179d60d37a5c5671b183`.
The training manifest fixes the category counts, ordering inputs, packing
rules, and output destination so the run cannot silently switch datasets.

### Main training run

| Setting | Value |
| --- | --- |
| Sequence length | 768 tokens |
| Effective batch | 2 sequences per optimizer update |
| Optimizer steps | 1,589 |
| Warmup steps | 47 |
| Warmup share | 2.96% of optimizer steps |
| Effective sequence slots | 3,178 |
| Maximum scheduled token positions | 2,440,704 |
| Peak learning rate | `3e-5` |
| Final training weights SHA-256 | `b28a1744927aa2d90f947d64163a1106136db4c18e1e94784d8adcd7639a3649` |
| General-route package SHA-256 | `0915944be328c0ebef2087a6007cfe4f046c5b83709d5929f2504da51c15fd15` |

The completed run retained a receipt binding the dataset, configuration, final
weights, and exported package. A runtime smoke test then loaded the exact
package used by the general route before the 48-case diagnostic.

### Tool specialization

The structured-tool continuation was selected from the pinned 6,750-record
source. Benchmark prompts were excluded from this selection:

| Tool continuation group | Records | Share |
| --- | ---: | ---: |
| Native tool-call dialogues | 600 | 55.05% |
| Agentic replay | 240 | 22.02% |
| English replay | 150 | 13.76% |
| Russian replay | 100 | 9.17% |
| **Total** | **1,090** | **100.00%** |

These records were packed into 992 sequences of at most 512 tokens. The
selection SHA-256 is
`2a9ecf6254ba2bb2da46fbda7aebf53d211157a0a1a04c57dd5814236ac611b7`.
General English, Russian, and agentic replay remain in the mixture to reduce
specialist overfitting. Tool output is checked against declared names and JSON
schemas, with one bounded repair attempt before failure.

Native tool dialogues and agentic replay make up 77.07% of this specialist
selection. The remaining 22.93% is English and Russian replay retained as a
language-regression buffer. Packing converts 1,090 records into 992 sequences,
with a maximum packed capacity of 507,904 token positions at 512 positions per
sequence.

### Capability isolation

The final system keeps ordinary language, code, and tool behavior on explicit
routes. The general route handles Russian, English, text quality, and ordinary
dialogue. The code route is selected only for programming requests because it
performs better on code while reducing prose quality. The tools route is
available only when the client supplies a declared tool schema.

This isolation prevents specialist tuning from becoming the default response
policy. Every routed response can expose the selected route and artifact hash,
and a requested specialist route fails closed if its package is missing.

### Refusal-reduction stage

A dedicated refusal-reduction stage ran before the final multi-capability
continuation. It targeted unjustified refusals on lawful, benign, and
open-topic requests while preserving separate client-side permission checks
for tools and external actions.

The interrupted training session did not retain the separate SFT-only versus
refusal-reduction ablation receipt or the expanded open-topic report. The card
therefore reports the retained 48-case refusal measurement and does not claim
that every possible prompt receives an answer.

### Evaluation and promotion controls

- The held-out diagnostic uses 48 fixed cases, eight per capability.
- Generation is fixed at temperature 0, seed 4242, and a 512-token output
  limit.
- Scores are recomputed from raw responses instead of copied from a summary.
- Missing cases, duplicate cases, truncated answers, forbidden terms, and
  artifact mismatches invalidate the affected evidence.
- A specialist package is not allowed to replace the general route without a
  full language and agentic regression check.

The training receipts support the routed candidate described in the benchmark
section. The standalone public GGUF remains a separately hashed artifact until
the routed packages are merged, signed, and published as one release.

## Install

### One-line installation

macOS on Apple Silicon:

```sh
curl -fsSL https://huggingface.co/metaflora/incubus/resolve/main/install.sh | sh
```

Text, voice, and vision:

```sh
curl -fsSL https://huggingface.co/metaflora/incubus/resolve/main/install.sh | sh -s -- --with-voice --with-vision
```

Enable vision on an existing installation without replacing the text weights:

```sh
incubusctl update --with-vision
```

The bootstrap is pinned to an immutable installer archive and verifies its
SHA-256 checksum before execution. Linux, Intel macOS, and Windows automatic
installation will be published only after signed runtime artifacts pass the
same release checks.

### OpenCode / open-source IDEs

The installer starts a loopback-only OpenAI-compatible service on the local
machine and adds Incubus to the current OpenCode configuration. Restart
OpenCode after installation, then select
`metaflora-incubus/metaflora-incubus-v1`.
An installation made with `--with-vision` also declares image input to
OpenCode and starts the same local endpoint with the signed native
`mmproj-F16` vision projector. Attach PNG or JPEG files through OpenCode; text-only
installations do not advertise image support.

| Field | Value |
| --- | --- |
| Provider ID | `metaflora-incubus` |
| Display name | `Metaflora Incubus v1` |
| Base URL | `http://127.0.0.1:18991/v1` |
| API key | leave empty, or use `local` if the client requires a value |
| Model ID | `metaflora-incubus-v1` |
| Model display name | `Metaflora Incubus v1` |
| Extra headers | none |

If OpenCode was installed after Incubus, register the provider without
downloading the model again:

```sh
incubusctl integrate opencode
```

Check the service before connecting an IDE:

```sh
incubusctl status
curl http://127.0.0.1:18991/v1/models
```

Run a one-shot prompt directly from the terminal:

```sh
incubusctl run "Explain this repository"
```

Use `incubusctl start`, `incubusctl stop`, and `incubusctl logs` to manage the
background service.

The same Base URL and Model ID work with open-source IDEs and local agent
clients that accept a custom OpenAI-compatible endpoint, including Continue,
Cline, Roo Code, Zed, Aider, and Open WebUI. Keep the service on `127.0.0.1`;
it has no API authentication because it is intended for local single-user use.

Opening links belongs to the client-side agent layer. In OpenCode, enable its
browser or web-fetch tool; the client retrieves the page and supplies its
contents to Incubus. The local runtime does not fetch arbitrary URLs by itself.

## Quickstart

### Ollama

Download `metaflora-incubus-v1.gguf` and create a `Modelfile`:

```dockerfile
FROM ./metaflora-incubus-v1.gguf
PARAMETER num_ctx 8192
```

```sh
ollama create metaflora-incubus:v1 -f Modelfile
ollama run metaflora-incubus:v1
```

### llama.cpp

```sh
llama-server \
  -m ./metaflora-incubus-v1.gguf \
  --host 127.0.0.1 \
  --port 11435 \
  -c 8192
```

The server exposes an OpenAI-compatible API at
`http://127.0.0.1:11435/v1`.

### Manually started llama.cpp clients

| Field | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:11435/v1` |
| API key | empty, or `local` if the client requires a value |
| Model ID | `metaflora-incubus-v1` |

Keep the endpoint bound to `127.0.0.1`. Public network exposure needs
authentication, request limits, and a separate security review.

## Agentic and tool usage

Agentic use requires a tool-enabled client. The model emits structured tool
calls; the client retains control over filesystem, browser, shell, and network
permissions.

Suitable workflows include:

- structured tool selection and JSON arguments;
- search tasks where a client supplies a search tool and returns sources;
- code generation, explanation, and review;
- local Russian and English drafting;
- private single-user inference.

The model itself has no network access. Every external action comes from the
client and its configured tools.

## Benchmark Results

### Routed release candidate

The latest 48-case diagnostics use a frozen general route plus isolated code
and tool routes. Each request is scored on the route intended for that
capability. The results are bound to the following artifacts:

- General route artifact:
  `0915944be328c0ebef2087a6007cfe4f046c5b83709d5929f2504da51c15fd15`
- Code and tool route artifact:
  `d066766d4bbd0346b8a4a02a35d752e39ff7f70b064b71272d6fa43bdb9be526`

| Capability | Route | Score (/100) |
| --- | --- | ---: |
| Agentic workflows | General | 100.00 |
| Code | Code | 95.83 |
| Tool use | Tools | 95.00 |
| English | General | 87.50 |
| Russian | General | 81.25 |
| Text quality | General | 81.25 |

These are direct diagnostics from the completed candidate runs. The code
score comes from the isolated code route; enabling that route for ordinary
prose would reduce English and text quality. The complete routed regression
run and production signature are still pending, so these values are not
presented as scores for the standalone GGUF download.

### Standalone public GGUF verification

The currently downloadable Q5_K_M file has SHA-256
`df850ca6f8d47b2d92db99fb623a36e9d35f3ad7737a588e7ff562ac5229b3fc`.
Its older signed receipt remains available under [`reports/`](reports/) for
artifact verification. It must not be mixed with the routed candidate scores
above.

### Evaluation setup

The deterministic suite contains 48 fixed cases, with eight cases per
capability. Generation uses temperature 0, seed 4242, a 512-token output
limit, and the declared runtime.

The final capability score is:

`100 × sum(case scores) / 8`

### Scoring

- Code, English, Russian, and text-quality cases list required answer elements.
  Each score is the fraction of required elements found in the normalized
  response.
- Tool and agentic cases award 60% for the exact tool name and 40% for exact
  JSON arguments.
- Refusals, forbidden terms, truncated responses, duplicate cases, missing
  cases, and mismatched artifact receipts invalidate or zero the affected
  evidence.
- Aggregates are recomputed from raw model responses.

These results describe the Incubus release suite. They are not directly
equivalent to MMLU-Pro, GPQA-Diamond, LiveCodeBench, BFCL, SWE-bench,
Terminal-Bench, or TAU2.

The standalone GGUF has signed evidence under [the published `reports/` directory](https://huggingface.co/metaflora/incubus/tree/main/reports). The
routed candidate measurements are diagnostic until the production signing run
is complete.

## Open local operation and refusal behavior

The local release has no mandatory cloud moderation service, remote policy API,
hidden server-side prompt filter, or vendor-controlled allowlist. Prompts and
generated text remain on the user's machine unless the selected client invokes
an external service. The model can operate offline after installation.

The earlier signed standalone run observed a 0% refusal rate on its 48 fixed
cases. This is a bounded measurement, not a promise that every possible prompt
receives an answer. Learned refusal patterns can still appear, and generated
answers can be incorrect. Applications may add their own policies at the
client boundary without changing the local artifact.

## Voice and vision

Voice and vision are optional signed packages. They remain separate from the
main GGUF so text-only users keep the compact primary download.

Vision uses the matching native `mmproj-F16` projector and is connected to the
OpenAI-compatible image-input route when installed with `--with-vision`.

Voice contains a compact 0.6B Q8_0 speech-recognition encoder and its matching
audio projector. The
installer verifies and retains this package independently. The OpenCode audio
bridge remains separate from the main image/text endpoint; the card does not
claim direct microphone or audio-attachment support until that bridge is
released.

## Processing long contexts

The release runtime is validated at 8,192 tokens. Larger contexts consume more
memory and require separate quality evaluation; they are not part of the v1
release claim.

## Best practices

- Use low sampling temperature for code and structured tool arguments.
- Validate tool names and arguments against a declared schema.
- Review generated code and tool actions before execution.
- Give agent clients the minimum required filesystem and network permissions.
- Keep the local endpoint on `127.0.0.1`.

## Limitations

- The routed candidate is not yet available as one merged standalone GGUF.
- The code route is intentionally isolated because it reduces ordinary
  English and text quality when used as the default route.
- Routed candidate diagnostics are not official BFCL, SWE-bench,
  Terminal-Bench, TAU2, LiveCodeBench, GPQA-D, HLE, or MMLU-Pro scores.
- Outputs may contain factual errors, broken code, invalid arguments, or
  invented citations.
- Search quality depends on the connected search tool and its source coverage.
- Other languages are not covered by the v1 release claim.
- Local execution does not make a tool-enabled client harmless.
- The model is not a substitute for professional medical, legal, financial, or
  safety-critical review.

## Artifact integrity

| Artifact | SHA-256 |
| --- | --- |
| `metaflora-incubus-v1.gguf` | `df850ca6f8d47b2d92db99fb623a36e9d35f3ad7737a588e7ff562ac5229b3fc` |
| `incubusctl-darwin-arm64.tar.gz` | `3582a04a8097787ad73ecfa64ae458083fd5c6188e821f90eef88005d217f79a` |
| Voice package | `32c051da39784ef5dea000df4d263f5e4b8d50631207a896c1961e847a22b818` |
| Vision package | `5f9ba0fb6af356b60e4082c2747b720f705890d05cdd6dfce8fd89a890b9d935` |

The repository includes an installer manifest, detached signature, signed
evaluation receipts, raw outputs, runtime information, and the legal notices
required for distribution.

## Reproducibility

- Model revision: `573eb9e077a833f676740c7ba2dca0b34003297d`
- Evaluation case-bank SHA-256:
  `9f18aba6bed35a1165cb5015ab10302f6a219c10ea1e564838a77cc3bcd75d49`
- Evaluation runtime SHA-256:
  `ed49acd23b9f537391a97fd3270708cb4cd3240165a57be3d77bfc6bd4fd806`
- Seed: 4242
- Temperature: 0

Required attribution and license terms are preserved in
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES).

## Repository implementation

This repository is the engineering source for the published model system. It contains the release builder, the bounded local installer, pinned benchmark catalogues, evidence generation, package signing, local runtime integration, and the tests that guard each part of the release path.

| Path | What it contains |
| --- | --- |
| `src/metaflora_incubus/` | Model build orchestration, benchmark runners, publication logic, release signing, and local runtime code |
| `benchmarks/` | Fixed evaluation inputs, baseline pins, and head-to-head run plans |
| `installer/` | Install, update, recovery, uninstall, and local-service integration |
| `release/templates/huggingface/` | Templates for the signed Hugging Face release bundle |
| `docs/product/` | Capability contract and release scope |
| `docs/quality/` | Weakness matrix, tool quality, and agentic smoke documentation |
| `docs/release/` | Packaging, integrity, evidence, and release-gate documentation |
| `notebooks/` | Reproducible Kaggle and benchmark notebooks |
| `tests/` | Unit, integration, installer, release, and evidence tests |

Start with the [product capability contract](docs/product/incubus-v1-capability.md), [local installation notes](docs/install/local.md), [weakness matrix](docs/quality/weakness-matrix.md), and [head-to-head launch plan](benchmarks/head-to-head-v1-launch-plan.md).

## Developer verification

Create a Python 3.11 environment and install the repository in editable mode:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the repository checks:

```sh
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pip check
git diff --check
```

The test suite checks implementation contracts. Public capability scores require raw model responses and signed evidence bound to an exact model artifact; unit tests do not substitute for those measurements.
