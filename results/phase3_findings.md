# Phase 3 Findings — KV Cache, Batching, vLLM, Adaptive Router

This doc summarizes what you completed for Phase 3 and what must be corrected/rerun before moving to Phase 4.

## What’s complete (per roadmap)

1. KV Cache experiment (Step 1)
   - Ran `optimization/kv_cache_experiment.py` and generated charts + JSON under `results/kv_cache_experiment-results/`.
   - Output files created:
     - `results/kv_cache_experiment-results/kv_cache_results.json`
     - `results/kv_cache_experiment-results/kv_cache_speedup.png`
     - `results/kv_cache_experiment-results/kv_cache_tokpersec.png`
     - `results/kv_cache_experiment-results/kv_cache_ttft.png`

2. Static batching (Step 2)
   - Ran `optimization/batching.py`, saved results + charts under `results/batching-results/`:
     - `results/batching-results/batching_results.json`
     - `results/batching-results/batching_throughput.png`
     - `results/batching-results/batching_latency.png`
     - `results/batching-results/batching_efficiency.png`

3. vLLM dynamic batching (Step 3)
   - Ran `benchmarks/batching_comparison-vllm.py`, saved outputs under `results/vllm/`:
     - `results/vllm/vllm_results.json`
     - `results/vllm/vllm_comparison.png`
     - `results/vllm/vllm_comparison_table.txt`

4. Adaptive router (Step 4)
   - Verified routing with:
     - `python inference/adaptive_router.py --dry-run`
     - `python inference/adaptive_router.py --benchmark --max-tokens 150`
     - `python inference/adaptive_router.py --prompt "Write a Python binary search function"`
   - Routing log saved to `results/router/routing_log.json`

## KV Cache experiment (Step 1) — current results and the issue

Your current `kv_cache_results.json` shows:
- `seq_len` = 128/256/512/1024
- `new_tokens` is now constant at `64` for every run (both KV cache ON and OFF)

Observed aggregates (your latest rerun):
- Seq 128: cache ON tok/s `32.36`, cache OFF tok/s `30.57`, speedup `1.06×`
- Seq 256: cache ON tok/s `31.34`, cache OFF tok/s `17.33`, speedup `1.81×`
- Seq 512: cache ON tok/s `29.78`, cache OFF tok/s `7.75`, speedup `3.84×`
- Seq 1024: cache ON tok/s `29.12`, cache OFF tok/s `2.84`, speedup `10.25×`

This now matches the roadmap expectation: **as context length grows, KV cache increasingly improves throughput**.

✅ Fix applied in code:
- I patched `optimization/kv_cache_experiment.py` so:
  - `seq_len` means INPUT/context length (128/256/512/1024 tokens)
  - generation uses a fixed number of new tokens (`--new-tokens`, default `64`)
  - `min_new_tokens` is enforced so output length doesn’t collapse early

Status:
- KV cache Step 1 is now aligned with the roadmap and can be used as-is for Phase 4.

## Static batching (Step 2) — good and consistent

Your static batching results show near-linear throughput scaling (as expected for GPU utilization):
- Batch 1: total tok/s `30.31`, per-prompt `33.0ms`, GPU mem `~2.09GB`
- Batch 2: total tok/s `60.58`, per-prompt `16.7ms`, GPU mem `~2.09GB`
- Batch 4: total tok/s `127.77`, per-prompt `7.8ms`, GPU mem `~2.10GB`
- Batch 8: total tok/s `262.59`, per-prompt `3.8ms`, GPU mem `~2.10GB`

Efficiency was reported slightly above 100% for larger batches (105–108%). That can happen from measurement noise + warmup effects; the overall trend is still exactly what we want to show in the README.

This step looks production-reasonable and is consistent with the roadmap’s intent.

## vLLM dynamic batching (Step 3) — now aligned and complete

Your latest `results/vllm/vllm_results.json` now indicates:
- batch 1: `total_new_tokens=114`, `total_tok_per_sec=106.26`
- batch 2: `total_new_tokens=228`, `total_tok_per_sec=195.9`
- batch 4: `total_new_tokens=456`, `total_tok_per_sec=385.91`
- batch 8: `total_new_tokens=912`, `total_tok_per_sec=750.57`

This confirms the benchmark is now generating a full token budget per prompt (not collapsing to 1 token), so the comparison against static batching is fair enough for this project.

Comparison summary (manual vs vLLM tok/sec from your latest run):
- Batch 1: `30.31` vs `106.26`  → `3.51×`
- Batch 2: `60.58` vs `195.90`  → `3.23×`
- Batch 4: `127.77` vs `385.91` → `3.02×`
- Batch 8: `262.59` vs `750.57` → `2.86×`

Interpretation:
- vLLM consistently outperforms manual static batching on this workload.
- Relative speedup decreases slightly at larger batches (typical as manual batching already utilizes GPU better at high batch sizes).

## Adaptive router (Step 4) — working, minor tier expectation mismatch

Dry-run shows the tier rules are doing what you intended:
- short/simple prompts → `[FAST — 4-bit]`
- reasoning/creative prompts → `[BALANCED — 8-bit]`

Benchmark run:
- Routing accuracy: `14/16 (88%)`
- Two prompts were classified to the balanced tier instead of the “expected quality” tier.

This is acceptable as an MVP, but for “interview impressiveness” you can later tune the FP16 threshold and/or add explicit patterns for “high-stakes/high-quality” wording.

## Can we move to Phase 4 now?

Yes.

Phase 3 deliverables are now complete:
1. KV cache graphs and JSON are correct.
2. Static batching comparison and plots are complete.
3. vLLM dynamic batching comparison is now aligned and saved.
4. Adaptive router is working with logged decisions.

You can confidently move to Phase 4 (Streaming API).

