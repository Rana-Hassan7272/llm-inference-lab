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
- `new_tokens` stayed constant at `14` for every run (both KV cache ON and OFF)

Observed aggregates (from your logs / `kv_cache_results.json`):
- Seq 128: cache ON tok/s `25.68`, cache OFF tok/s `31.18`, speedup `0.82×`
- Seq 256: cache ON tok/s `34.02`, cache OFF tok/s `31.13`, speedup `1.09×`
- Seq 512: cache ON tok/s `34.86`, cache OFF tok/s `31.37`, speedup `1.11×`
- Seq 1024: cache ON tok/s `28.63`, cache OFF tok/s `28.72`, speedup `1.00×`

This pattern is not a “clean KV cache scaling curve” because the experiment is currently not holding the generation length constant (the model stops early at ~14 tokens).

Additionally, the implementation you ran treated “sequence length” as `max_new_tokens` rather than true *context length*. The roadmap asks for *sequence length* scaling (context), so the interpretation doesn’t fully match the requirement.

✅ Fix applied in code:
- I patched `optimization/kv_cache_experiment.py` so:
  - `seq_len` means INPUT/context length (128/256/512/1024 tokens)
  - generation uses a fixed number of new tokens (`--new-tokens`, default `64`)
  - `min_new_tokens` is enforced so output length doesn’t collapse early

Required rerun:
- Rerun KV cache Step 1 on Colab GPU using the patched script, then regenerate the same `results/kv_cache_experiment-results/*` charts/JSON.

## Static batching (Step 2) — good and consistent

Your static batching results show near-linear throughput scaling (as expected for GPU utilization):
- Batch 1: total tok/s `30.31`, per-prompt `33.0ms`, GPU mem `~2.09GB`
- Batch 2: total tok/s `60.58`, per-prompt `16.7ms`, GPU mem `~2.09GB`
- Batch 4: total tok/s `127.77`, per-prompt `7.8ms`, GPU mem `~2.10GB`
- Batch 8: total tok/s `262.59`, per-prompt `3.8ms`, GPU mem `~2.10GB`

Efficiency was reported slightly above 100% for larger batches (105–108%). That can happen from measurement noise + warmup effects; the overall trend is still exactly what we want to show in the README.

This step looks production-reasonable and is consistent with the roadmap’s intent.

## vLLM dynamic batching (Step 3) — current comparison is not fair yet

Your `results/vllm/vllm_results.json` indicates:
- `total_new_tokens` equals the batch size exactly:
  - batch 1 → 1 new token
  - batch 2 → 2 new tokens
  - batch 4 → 4 new tokens
  - batch 8 → 8 new tokens

That means the vLLM run generated only ~1 token per prompt, so it is not comparable to manual batching where you benchmarked full generation length.

✅ Fix applied in code:
- I patched `benchmarks/batching_comparison-vllm.py` so it computes prompt lengths and budgets vLLM’s `SamplingParams(max_tokens=prompt_len + max_new_tokens)` and increases `max_model_len` accordingly.

Required rerun:
- Rerun the vLLM comparison on Colab GPU using the patched script, then regenerate `results/vllm/*`.

## Adaptive router (Step 4) — working, minor tier expectation mismatch

Dry-run shows the tier rules are doing what you intended:
- short/simple prompts → `[FAST — 4-bit]`
- reasoning/creative prompts → `[BALANCED — 8-bit]`

Benchmark run:
- Routing accuracy: `14/16 (88%)`
- Two prompts were classified to the balanced tier instead of the “expected quality” tier.

This is acceptable as an MVP, but for “interview impressiveness” you can later tune the FP16 threshold and/or add explicit patterns for “high-stakes/high-quality” wording.

## Can we move to Phase 4 now?

Not yet.

Phase 3 is functionally done (you produced KV cache, batching, vLLM, and a working adaptive router), but **the KV cache Step 1 and vLLM Step 3 measurements are not aligned with the roadmap definition** due to fixed-length/context-length issues.

To safely move to Phase 4, rerun:
1. `optimization/kv_cache_experiment.py` (Step 1) with the patched context-length logic.
2. `benchmarks/batching_comparison-vllm.py` (Step 3) so vLLM generates the intended number of new tokens per prompt.

After reruns, we should be able to produce the Phase 3 graphs/JSON that you can confidently reference in Phase 4 (streaming API + routing).

