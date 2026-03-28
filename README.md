# LLM Inference Lab — End‑to‑End Systems Project

A production‑minded journey from single‑GPU experiments to a deployable LLM inference stack with benchmarks, streaming API, dynamic batching (vLLM), load testing, tracking, and a live dashboard.

- Frontend (Vercel): https://llm-inference-lab.vercel.app/
- Backend API (Render): https://llm-inference-lab.onrender.com/


## Project Highlights

- Adaptive routing across precision tiers (4‑bit / 8‑bit / FP16) for quality vs latency
- Streaming FastAPI server with SSE token streaming and full metrics (TTFT, TPS, VRAM)
- ModelManager for lazy loading, tier caching, and VRAM tracking
- Reproducible optimization experiments: quantization, KV cache, static vs dynamic batching
- vLLM dynamic batching benchmark with forced generation length to ensure fair comparisons
- Automated load‑testing sweeps (Locust) with summary artifacts
- MLflow integration for experiment tracking (local file store)
- Production deploys: API (Docker → Render), Dashboard (Vercel)


## Repo Structure

```
api/
  app.py                      # FastAPI server (streaming, routing, benchmarks)
benchmarks/
  runner.py                   # Orchestrates Phase 5 experiments
  batching_comparison-vllm.py # vLLM vs manual batching
  locustfile.py               # Load test behavior (POST /generate, GET /health)
  load_test_runner.py         # Automated Locust sweeps + artifacts
  mlflow_integration.py       # MLflow logging (file:// tracking URI on Windows-safe path)
  merge_scored_results.py     # Merge *_scored.json → canonical *_results.json
dashboard/
  src/App.jsx                 # Dashboard UI (comparison table, charts, routing log)
  public/data/                # Deployed artifacts (dashboard_bundle.json, load_test_summary.json)
inference/
  model_manager.py            # Tiered (4‑bit/8‑bit/FP16) lazy loader with VRAM tracking
  adaptive_router.py          # Heuristics to choose tier by prompt complexity
optimization/
  kv_cache_experiment.py      # KV‑cache scaling vs context length (fixed new tokens)
  batching.py                 # Manual/static batching baseline
results/
  ...                         # All artifacts (see phase sections below)
Dockerfile
docker-compose.yml            # api + mlflow + dashboard (local dev)
requirements.txt
README.md
```


## Phase 1–2 — Setup and Baselines

- Environment setup (venv, requirements)
- Baseline generation and quick metrics collection

Key idea: establish a stable baseline on CPU/GPU and make sure tokenization, generation, and basic timing work (noisy results are acceptable as long as the workflow is solid).


## Phase 3 — Optimization Experiments

Scope: KV cache scaling, static batching, vLLM dynamic batching, adaptive router.

Source of truth: `results/phase3_findings.md`

### KV Cache (Step 1)
- Experiment: `optimization/kv_cache_experiment.py`
- Fixed: interpret `seq_len` as context length and enforce constant new tokens (`64`) via `min_new_tokens`.
- Artifacts:
  - `results/kv_cache_experiment-results/kv_cache_results.json`
  - `.../kv_cache_speedup.png`, `.../kv_cache_tokpersec.png`, `.../kv_cache_ttft.png`
- Findings (example from latest run):
  - Speedup grows with context: ~1.06× @128, 1.81× @256, 3.84× @512, 10.25× @1024

### Static Batching (Step 2)
- Experiment: `optimization/batching.py`
- Artifacts:
  - `results/batching-results/batching_results.json`
  - `.../batching_throughput.png`, `.../batching_latency.png`, `.../batching_efficiency.png`
- Findings (from latest run):
  - Near‑linear throughput scaling; total tok/s: 30.31 → 60.58 → 127.77 → 262.59 (batch 1→8)

### vLLM Dynamic Batching (Step 3)
- Experiment: `benchmarks/batching_comparison-vllm.py`
- Fixes applied:
  - Count new tokens correctly from `token_ids`
  - Force generation length per prompt (`min_tokens=max_new_tokens`, `ignore_eos=True`)
- Artifacts:
  - `results/vllm/vllm_results.json`
  - `results/vllm/vllm_comparison.png`
  - `results/vllm/vllm_comparison_table.txt`
- Findings (example):
  - Speedups vs manual batching on total tok/s:
    - Batch 1: 106.26 vs 30.31 → 3.5×
    - Batch 8: 750.57 vs 262.59 → 2.86×

### Adaptive Router (Step 4)
- Module: `inference/adaptive_router.py`
- Behavior: short/simple → `fast` (4‑bit), reasoning/creative → `balanced` (8‑bit), can force `quality` (FP16)
- Routing evidence:
  - `results/router/` + API routing log: `results/routing_log.jsonl`


## Phase 4 — Production‑style API

Endpoints:
- `POST /generate` — blocking generation with routing
- `POST /generate/stream` — SSE token streaming
- `GET /benchmark/{model_tier}` — quick 3‑prompt benchmark
- `GET /health`, `GET /status` — system/model status
- `GET /router/explain`, `GET /routing-log` — routing insights

Key files:
- `api/app.py`, `inference/model_manager.py`

Confirmed on GPU (Colab T4) and CPU (with FP16 fallback). See `results/phase4_findings.md` for detailed notes and caveats.


## Phase 5 — Automation, Load Testing, Tracking, Dashboard

- Step 1: `benchmarks/runner.py` orchestrates experiments (skips CUDA‑only on CPU, sets UTF‑8 on Windows).
- Step 2: Load testing
  - Behavior: `benchmarks/locustfile.py`
  - Runner: `benchmarks/load_test_runner.py`
  - Example successful run (Colab → API localhost):
    - `users=1 → rps≈1.16, p95=330ms`
    - `users=5 → rps≈4.27, p95=910ms`
    - `users=10 → rps≈4.61, p95=2000ms`
    - `users=20 → rps≈4.61, p95=3800ms`
  - Artifacts:
    - `results/load-testing/run_*/load_test_summary.json`
    - `results/load-testing/run_*/throughput_trend.csv`
- Step 3: MLflow logging
  - `benchmarks/mlflow_integration.py`
  - Windows path fix to ensure `file://` tracking URI
- Step 4: React dashboard (Vite)
  - `dashboard/src/App.jsx`
  - Loads `public/data/dashboard_bundle.json` and `load_test_summary.json` (or `VITE_LATENCY_JSON_URL`)
  - Visuals: comparison table; memory‑vs‑quality scatter; latency/throughput dual‑axis; routing log


## Phase 6 — Docker & Deployment

- Dockerfile for API (uvicorn launcher, healthcheck)
- docker-compose (local): api + mlflow + dashboard
- Cloud deploys:
  - API on Render (Docker)
  - Dashboard on Vercel (Vite build)

Links:
- Frontend (Vercel): https://llm-inference-lab.vercel.app/
- Backend API (Render): https://llm-inference-lab.onrender.com/


## Key Results (Examples)

### Load Test Summary (local/Colab → API localhost)
From `results/load-testing/run_20260327_164725/load_test_summary.json`:

| Users | Requests/s | Avg (ms) | P95 (ms) | Fail ratio |
|------:|-----------:|---------:|---------:|-----------:|
|     1 |       1.16 |   272.82 |      330 |       0.00 |
|     5 |       4.27 |   495.25 |      910 |       0.00 |
|    10 |       4.61 |  1258.01 |     2000 |       0.00 |
|    20 |       4.61 |  2931.82 |     3800 |       0.00 |

### vLLM vs Manual Batching (Phase 3)
From `results/vllm/vllm_results.json` and `results/batching-results/batching_results.json`:

- Batch 1: manual ~30.31 tok/s vs vLLM ~106.26 tok/s → ~3.5×
- Batch 8: manual ~262.59 tok/s vs vLLM ~750.57 tok/s → ~2.86×

### KV Cache Scaling (Phase 3)
From `results/kv_cache_experiment-results/kv_cache_results.json`:

- Speedup grows with context: ~1.06× → 1.81× → 3.84× → 10.25× (128→1024)

### Representative Inference Report (GPU, T4)
Using `/generate` (forced tier optional), example (TinyLlama/TinyLlama‑1.1B‑Chat‑v1.0):

- Prompt: `To be, or not to be,`
- max_new_tokens: `64`, temperature: `0.8`, top_p: `0.9`, top_k: `40`
- Time: `~4.134 s` | Tokens: `64` | TPS: `~15.48/s`
- Device: `GPU (Tesla T4)` | Memory used: `~1.94 GB`
- Source: API `/generate`


## How to Run — Local (Dev)

Backend (Python 3.11 recommended):

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Dashboard:

```bash
cd dashboard
npm ci
npm run dev   # http://127.0.0.1:5173
```


## How to Run — Docker (Local compose)

```bash
docker compose up -d --build
```

Services:
- api: http://127.0.0.1:8000
- mlflow: http://127.0.0.1:5000
- dashboard: http://127.0.0.1:5173


## How to Deploy — Cloud

API (Render):
- New → Web Service → Connect GitHub → Runtime: Docker
- Env vars: `MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0` (and `HF_TOKEN` if needed)
- Healthcheck: `/health`
- Render will use `Dockerfile` (EXPOSE 8000, uvicorn CMD)

Dashboard (Vercel):
- New Project → Import GitHub
- Root: `dashboard`
- Build: `npm ci && npm run build`
- Output: `dist`
- Data:
  - Commit artifacts to `dashboard/public/data/`:
    - `dashboard_bundle.json`
    - `load_test_summary.json` (or set `VITE_LATENCY_JSON_URL`)


## Notes on Output Quality and Tradeoffs

TinyLlama in lower‑precision tiers (4‑bit/8‑bit) can produce noisier outputs, especially with chat formatting and greedy decoding. This is an expected tradeoff and is documented in `results/phase4_findings.md`. It does not detract from the systems engineering value:

- Shows tiering tradeoffs (memory, TTFT, TPS, quality)
- Demonstrates routing decisions and production‑style serving

For showcase demos, prefer the `quality` tier (FP16) or tune prompts/decoding params.


## What to Include in Architecture Section (for your diagram)

When you generate the diagram (e.g., in Claude/Excalidraw), include:
- Client (Dashboard/CLI) → API Gateway (FastAPI, Uvicorn)
- Adaptive Router (heuristics; prompt length/intent → tier)
- ModelManager (tiered models: 4‑bit / 8‑bit / FP16; lazy load; VRAM tracking; per‑tier locks)
- Generation paths:
  - Blocking (`/generate`): TTFT + total latency + TPS measured
  - Streaming (`/generate/stream`): TextIteratorStreamer thread + SSE to client
- Experiment pipeline:
  - KV cache, static batching, vLLM scripts
  - MLflow logger (file store)
  - Locust load testing → artifacts (summary/trend CSV)
- Observability/data plane:
  - Routing log (JSONL) → surfaced to dashboard
  - Results folder → dashboard `public/data` artifacts
- Deployment:
  - Dockerized API (Render), Static dashboard (Vercel)


## Quick API Examples

Health:
```bash
curl -s https://llm-inference-lab.onrender.com/health
```

Blocking generation:
```bash
curl -s -X POST "https://llm-inference-lab.onrender.com/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello from Render","max_tokens":32,"temperature":0.0}'
```

Streaming (SSE):
```bash
curl -N -X POST "https://llm-inference-lab.onrender.com/generate/stream" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain recursion briefly","max_tokens":64,"temperature":0.0}'
```


## Credits

- Hugging Face Transformers/Accelerate
- vLLM
- FastAPI/Uvicorn
- Locust
- MLflow
- Recharts/Vite/React

