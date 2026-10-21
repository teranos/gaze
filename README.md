# qntx-gaze-plugin

Production LLM inference via llama.cpp with Metal acceleration. Fork of scry, stripped of signal extraction, nebula visualization, and observer samplers. Model in, tokens out.

## Build

Gaze speaks QNTX's plugin protocol: `plugin/grpc/protocol/{sigil,domain,llm}.proto` and `proto.cmake` there are compiled into the binary. `flake.nix` takes QNTX as a source input, pinned in `flake.lock`, and hands CMake its path as `QNTX_SRC`.

```
nix build
```

Without nix, CMake wants a QNTX checkout:

```
make install QNTX_SRC=/path/to/QNTX
```

## Configuration

Gaze is its record on the node ([ADR-002](https://github.com/teranos/QNTX/blob/main/docs/adr/ADR-002-plugin-configuration.md)): `plugins_add https://github.com/teranos/gaze`, `PUT /api/plugins/gaze/config` with the keys below, `plugins_enable`. The record's `build.*` keys for this repository: `build.core` `github.com/teranos/gaze@main`, `build.command` `nix build`, `build.output` `result/bin/qntx-gaze-plugin`.

| Key | |
|---|---|
| `models` | A JSON array of GGUF paths: `["/path/to/qwen-3b-Q4_K_M.gguf","/path/to/classifier-Q8.gguf"]`. Gaze reads each model's name from its GGUF metadata (`general.name`) and advertises it under that name. Callers address models by name. Gaze does not choose. |
| `model_path` | One GGUF path, read when `models` is empty. |
| `n_ctx` | Max context window (cap). Models use their native size if smaller. 2048 when unset. |
| `log_level` | `error`, `warn`, `info` or `debug`. `info` when unset. |

## Contract

- **Gaze loads, gaze serves.** Every GGUF in `models` is loaded at startup and held in memory. Each gets its own llama.cpp context and mutex.
- **Callers choose.** The `model` field in the gRPC request must match an advertised name. Gaze does not pick models — the caller does.
- **Names come from the model.** `general.name` GGUF metadata is the advertised name. Falls back to filename minus `.gguf` if metadata is missing.
- **Context is per-model.** Each model gets its native context length (from GGUF metadata), capped by `n_ctx` in config.
- **Sampler config is global.** Applies to all models. Override per-request if needed.
- **RAM is the limit.** Each model holds its weights + KV cache in memory. A 3B Q4 is ~2GB. Plan accordingly.

## Difference from scry

Scry instruments every token with signal data and renders a 3D nebula. Gaze is the production path — same llama.cpp engine, no instrumentation overhead. Tokens stream directly from the sampler to gRPC.

## Limitations

- **SINF** — Serial inference per model. Each model has its own context and mutex — different models run concurrently, but two requests to the same model still queue. True parallel inference on one model needs multiple contexts sharing weights.
- **NOAL** — No aliasing. Callers must use the exact name from GGUF metadata (e.g. "Qwen2.5 VL 3B Instruct"). No short names yet.
- **VDIR** — Vision mmproj must be in the same directory as the model and share its name prefix (e.g. `mmproj-Qwen2.5-VL-3B-Instruct-*.gguf` matches `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf`).
- **NDOC** — PDF only. DOCX, RTF, and other formats not supported.
- **IBP** — Image-based PDFs return empty text. OCR not supported.
- **UIG** — UI references scry. `/ws/llm` proxy and dynamic provider discovery in `llm-provider-glyph.ts` are implemented but not yet verified end-to-end.
- **SDR** — Shutdown race between gRPC teardown and llama.cpp destructor. Cosmetic log noise.

## Feature Requests

- **NMOD** — Gaze should refuse to start (or report unhealthy) when no models are configured. Currently it starts successfully with "no models loaded", which masks configuration errors and causes silent downstream failures for any plugin relying on the LLM service.
