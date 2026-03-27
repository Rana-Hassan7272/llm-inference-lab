# Phase 4 Findings — Streaming API

## Scope completed

Phase 4 Steps 1–4 are implemented and exercised:
- `api/app.py` with `/generate`, `/generate/stream`, `/benchmark/{model_tier}`
- Streaming via `StreamingResponse` (SSE-style token events)
- `inference/model_manager.py` for tiered loading/caching (`fast`, `balanced`, `quality`)
- Adaptive routing connected to generation path and routing decisions logged

## Runtime validation from your logs

### Server status
- `/health` is successful on Colab:
  - `status: ok`
  - `device: cuda`
  - `gpu_name: Tesla T4`

### Standard generation
- `/generate` works end-to-end (routing + model load + metrics).
- Example observed:
  - router selected `fast` (4-bit), request completed with timing/throughput metrics.

### Streaming generation
- `/generate/stream` returns token chunks in streaming format (`data: {"token": ...}`) and final done payload with metrics.
- This confirms token-by-token server streaming behavior is functioning.

### Routing + model tiers
- For short factual prompt, adaptive router correctly picks `fast`.
- With forced tiers:
  - `quality` produced coherent output patterns.
  - `balanced` produced output but with template/noisy artifacts in some runs.

## Quality observations (important for capstone positioning)

Observed output quality inconsistency in quantized tiers is expected with:
- TinyLlama as base model
- instruction/chat formatting sensitivity
- greedy/near-greedy decoding
- 4-bit/8-bit tradeoffs

This does **not** invalidate the capstone. The project goal is systems/inference engineering:
- routing decisions
- model tiering and memory-performance tradeoffs
- streaming API behavior
- benchmarkability and production architecture

Quality variance should be documented honestly as part of tradeoff analysis, not hidden.

## Recommended production notes

1. Keep current routing architecture and logging as-is (strong systems design signal).
2. For demos, prefer `quality` tier or improved prompt templates to avoid empty/noisy outputs.
3. Add explicit generation defaults for stability (temperature/top_p/repetition_penalty) in future iteration.
4. Keep benchmark metrics and routing logs in results for reproducibility.

## Deliverable status (Phase 4 so far)

- Step 1: done
- Step 2: done
- Step 3: done
- Step 4: done
- Step 5 (tests): added in `tests/test_api.py`

## Test run result (local)

`pytest -q tests/test_api.py` result:
- `4 passed`
- no failing tests

Warnings seen:
- Pydantic v2 deprecation: class-based `Config` in `GenerateRequest` should migrate to `ConfigDict`
- FastAPI deprecation: `@app.on_event("startup"/"shutdown")` should migrate to lifespan handlers

These are non-blocking deprecation warnings (current behavior is correct), but should be cleaned in a future polish pass.

