# Benchmark Results Summary

This document consolidates the key outcomes from the `results/` artifacts into one place for fast review.

## Snapshot

- **KV cache** gives very large speedups as context grows, reaching ~9.56x at 1024 tokens.
- **Static batching** scales throughput from `0.61` tok/s (batch 1) to `4.33` tok/s (batch 8), ~7.10x.
- **vLLM dynamic batching** strongly outperforms manual batching at all tested batch sizes.
- **Router quality** improved and is now measurable with a labeled evaluation set (`88.89%` accuracy, `0.8904` macro-F1).
- **Load testing** shows stable throughput plateau with 0% failures up to 20 concurrent users.

## 1) Quantization Ablation (from `results/ablation_table.json`)

| Tier | VRAM (GB) | Avg TTFT (ms) | Avg TPS (tok/s) | Auto Quality (1-5) | BLEU-1 (0-100) | BLEU Coverage (%) |
|---|---:|---:|---:|---:|---:|---:|
| 4-bit (fast) | 2.19 | 74 | 14.06 | 0.75 | 8.08 | 35.0 |
| 8-bit (balanced) | 2.16 | 152 | 7.16 | 0.25 | 1.90 | 35.0 |
| FP16 (quality) | 2.16 | 103 | 26.67 | 0.25 | 1.73 | 35.0 |

Interpretation:
- These values are useful for trend visibility, but BLEU coverage is partial and should be interpreted as a lightweight proxy.
- For strict comparison, regenerate all tier results in the same environment and re-run scoring.

## 2) KV Cache Experiment (from `results/kv_cache_experiment-results/kv_cache_results.json`)

| Context Length | Cache ON (tok/s) | Cache OFF (tok/s) | Speedup |
|---:|---:|---:|---:|
| 128 | 32.80 | 27.95 | 1.17x |
| 256 | 34.99 | 17.82 | 1.96x |
| 512 | 33.52 | 8.39 | 4.00x |
| 1024 | 29.15 | 3.05 | 9.56x |

Interpretation:
- KV cache impact grows superlinearly with prompt length.
- At long contexts, cache is essential for practical throughput.

## 3) Static Batching (from `results/batching-results/batching_results.json`)

| Batch Size | Total Throughput (tok/s) | Scaling vs Batch=1 |
|---:|---:|---:|
| 1 | 0.61 | 1.00x |
| 2 | 2.32 | 3.80x |
| 4 | 3.44 | 5.64x |
| 8 | 4.33 | 7.10x |

Interpretation:
- Throughput rises strongly with batch size.
- This is the key baseline for comparing vLLM dynamic batching.

## 4) vLLM Dynamic Batching (from `results/vllm/vllm_results.json`)

| Batch Size | vLLM Throughput (tok/s) |
|---:|---:|
| 1 | 107.01 |
| 2 | 195.03 |
| 4 | 384.55 |
| 8 | 755.13 |

Interpretation:
- vLLM is significantly faster than manual static batching in this artifact set.
- The gap reflects continuous/dynamic batching and optimized KV memory management.

## 5) Router Validation (from `results/router_eval_report.json`)

### Overall
- Accuracy: **88.89%** (40/45)
- Macro-F1: **0.8904**

### Per-tier Metrics
- Fast: precision `1.0000`, recall `0.9333`, F1 `0.9655`
- Balanced: precision `0.7778`, recall `0.9333`, F1 `0.8485`
- Quality: precision `0.9231`, recall `0.8000`, F1 `0.8571`

### Confusion Matrix (rows=true, cols=pred)

| True \ Pred | Fast | Balanced | Quality |
|---|---:|---:|---:|
| Fast | 14 | 1 | 0 |
| Balanced | 0 | 14 | 1 |
| Quality | 0 | 3 | 12 |

Interpretation:
- Router issue around never selecting `quality` was resolved.
- Remaining misses are mostly boundary cases between `balanced` and `quality`.

## 6) Load Testing (from `dashboard/public/data/load_test_summary.json`)

| Users | Generate RPS | Avg Latency (ms) | P95 Latency (ms) | Failure Ratio |
|---:|---:|---:|---:|---:|
| 1 | 1.16 | 272.82 | 330 | 0.0% |
| 5 | 4.27 | 495.25 | 910 | 0.0% |
| 10 | 4.61 | 1258.01 | 2000 | 0.0% |
| 20 | 4.61 | 2931.82 | 3800 | 0.0% |

Interpretation:
- Throughput plateaus around ~4.6 req/s while queueing latency increases with concurrency.
- Reliability remained strong (0% failures in this run).

## Notes for Reviewers

- Artifact locations:
  - Quantization/ablation: `results/ablation_table.json`
  - KV cache: `results/kv_cache_experiment-results/kv_cache_results.json`
  - Batching: `results/batching-results/batching_results.json`
  - vLLM: `results/vllm/vllm_results.json`
  - Router eval: `results/router_eval_report.json`
  - Load test summary: `dashboard/public/data/load_test_summary.json`
- Metrics reflect the committed artifacts and can vary by hardware/runtime.
