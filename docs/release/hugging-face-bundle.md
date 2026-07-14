# Hugging Face release bundle

The release directory under `release/templates/huggingface` is a staging
template for Metaflora Incubus v1. It deliberately contains unresolved tokens
and cannot be uploaded as a finished model repository.

## Required published files

The final Hugging Face repository contains:

| File | Requirement |
| --- | --- |
| `README.md` | Rendered model card with no template tokens |
| `metaflora-incubus-v1-q4.gguf` | Final release-gated GGUF artifact |
| `Modelfile` | Ollama import recipe for the local GGUF |
| `SHA256SUMS` | Checksums generated from the exact uploaded bytes |
| `release-manifest.json` | Release ID, model path, size, and SHA-256 |
| `release-manifest.sig` | Detached signature for the release manifest |
| `benchmark-report.json` | Measured results from the final GGUF |
| `benchmark-provenance.json` | Harness, commands, hardware, dataset revisions |
| `benchmark-provenance.sig` | Detached signature for benchmark provenance |
| `benchmark-cases.jsonl` | Exact prompts and stable case IDs used for the release run |
| `benchmark-raw.jsonl` | Exact responses, per-case scores and refusal labels |
| `benchmark-decision.json` | Signed release-gate decision for the measured GGUF |
| `benchmark-decision.sig` | Detached signature for the release-gate decision |
| `smoke-test.json` | Request and response from the final local artifact |
| `smoke-test.sig` | Detached signature for the smoke-test transcript |
| `THIRD_PARTY_NOTICES` | Complete legally required notices and licence texts |
| `LICENSE` | Distribution licence approved for the final artifact |

The release archive keeps the exact benchmark cases and raw outputs. Their
hashes, sample counts and artifact revision are bound to the signed provenance.
The public repository may keep large raw logs in an immutable release archive,
but the manifest must link that archive by URL and SHA-256.

## Render and validation rules

Release automation copies the `.tmpl` files into a clean staging directory,
replaces each
`${NAME}` token from signed build metadata, and writes files without the
`.tmpl` suffix. It then applies these checks:

1. The GGUF size is measured from disk and falls inside the approved v1 range.
2. SHA-256 values are recomputed after every file has reached its final form.
3. The release decision is `pass` and references the same model hash.
4. Every benchmark row comes from the final quantized artifact.
5. `README.md` contains neither `${...}` nor `NOT_MEASURED`.
6. `THIRD_PARTY_NOTICES` contains no legal-review marker and passes licence
   review.
7. The staged bundle contains no credentials, access tokens, private paths, or
   undeclared executable files.
8. A fresh Ollama import, llama.cpp launch, OpenAI-compatible API request, and
   signed installer run succeed against the staged GGUF.

The upload job stops on the first failed check. It must not create a public model
page with guessed scores, an unverified model file, or incomplete legal notices.

## Publication sequence

1. Freeze the release candidate and its benchmark manifest.
2. Run the complete benchmark and refusal suites on the final GGUF.
3. Pass licence review and render `THIRD_PARTY_NOTICES` plus `LICENSE`.
4. Render the model card and release manifest from measured artifacts.
5. Validate all hashes and run clean client smoke tests.
6. Create or update the Hugging Face model repository as a draft or private
   repository.
7. Upload immutable artifacts, read them back, and verify their hashes.
8. Make the repository public only after the downloaded GGUF passes the same
   smoke tests.
9. Put the verified model URL and revision into the signed installer manifest.

The public URL is recorded only after read-back verification. Until then,
documentation must state that no public checkpoint exists.

## Client support contract

The same GGUF artifact supports four entry paths:

- the signed Incubus installer, which manages the loopback service and OpenCode
  configuration;
- Ollama through `FROM ./metaflora-incubus-v1-q4.gguf`;
- llama.cpp through `llama-server`;
- any client that accepts an OpenAI-compatible base URL and model ID.

Client-specific packaging must never alter the model bytes without producing a
new filename, checksum, benchmark result, and release manifest.
