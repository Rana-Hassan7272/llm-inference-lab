"""
optimization/vllm_experiment.py
================================
Phase 3 — Step 3: vLLM Dynamic Batching Experiment

Compares manual static batching (your Phase 3 Step 2 results)
against vLLM's continuous/dynamic batching engine.

WHERE TO RUN:  Google Colab (free T4 GPU)
               vLLM requires Linux + CUDA. Does NOT run on CPU or Windows.

SETUP IN COLAB (run these cells first):
  !pip install vllm --quiet
  # Restart runtime after install, then run this script.

USAGE:
  python optimization/vllm_experiment.py
  python optimization/vllm_experiment.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  python optimization/vllm_experiment.py --compare-file batching_results.json

OUTPUTS:
  vllm_results.json           — raw vLLM metrics
  vllm_comparison.png         — vLLM vs manual batching chart
  vllm_comparison_table.txt   — plain text table for copy-paste into README
"""

import time
import json
import argparse
import statistics
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ── vLLM import with clear error message ──────────────────────────────────────
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  vLLM not installed.                                             ║
║                                                                  ║
║  Install in Colab:                                               ║
║    !pip install vllm                                             ║
║    Then: Runtime → Restart runtime                               ║
║    Then re-run this script.                                      ║
║                                                                  ║
║  Note: vLLM requires Linux + NVIDIA GPU.                         ║
║        It does NOT work on CPU or Windows.                       ║
╚══════════════════════════════════════════════════════════════════╝
""")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_SIZES     = [1, 2, 4, 8]
MAX_NEW_TOKENS  = 100
N_REPEATS       = 3

# Same prompt pool as batching.py so comparison is fair
PROMPT_POOL = [
    "What is photosynthesis? Explain briefly.",
    "Name three programming languages and their use cases.",
    "Explain Newton's first law of motion in simple terms.",
    "What is the difference between RAM and ROM?",
    "Describe what machine learning is in two sentences.",
    "What causes rainbows to form?",
    "Explain recursion in programming with a short example.",
    "What is the boiling point of water at sea level?",
    "Briefly describe what a neural network is.",
    "What is an API? Give a simple example.",
    "Who wrote Hamlet and when?",
    "What is the largest planet in our solar system?",
    "What is the Pythagorean theorem?",
    "Translate 'Good morning' into Spanish and French.",
    "What year did World War II end and why?",
    "Explain what the internet is in one paragraph.",
]


def get_batch_prompts(batch_size):
    return [PROMPT_POOL[i % len(PROMPT_POOL)] for i in range(batch_size)]


def print_header(text):
    bar = "═" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


# ══════════════════════════════════════════════════════════════════════════════
#  vLLM BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
def run_vllm_benchmark(llm, batch_sizes, max_new_tokens=MAX_NEW_TOKENS, max_prompt_len=0):
    """
    vLLM key difference: you submit ALL prompts at once.
    The engine schedules them using continuous batching internally —
    it fills GPU compute slots as individual sequences finish,
    never waiting for the slowest in a batch.
    """
    # Some vLLM versions interpret SamplingParams.max_tokens as TOTAL tokens
    # (prompt + generated). In your previous run it generated only ~1 token.
    # To force "max_new_tokens", we budget: max_tokens = max_prompt_len + max_new_tokens.
    sampling_params = SamplingParams(
        max_tokens=max_prompt_len + max_new_tokens,
        temperature=0,  # greedy = deterministic = fair comparison
    )
    results = {}

    for bs in batch_sizes:
        print_header(f"vLLM — Batch Size = {bs}")
        prompts = get_batch_prompts(bs)

        # Warmup
        llm.generate(
            get_batch_prompts(1),
            SamplingParams(max_tokens=max_prompt_len + 10, temperature=0),
        )

        run_data = []
        for i in range(N_REPEATS):
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            wall_time = time.perf_counter() - t0

            total_new_tokens = sum(
                len(o.outputs[0].token_ids) for o in outputs
            )
            total_tok_per_sec = total_new_tokens / wall_time if wall_time > 0 else 0
            per_prompt_ms     = (wall_time / bs) * 1000

            run_data.append({
                "total_tok_per_sec": round(total_tok_per_sec, 2),
                "per_prompt_ms":     round(per_prompt_ms, 1),
                "total_new_tokens":  total_new_tokens,
                "wall_time_ms":      round(wall_time * 1000, 1),
                "batch_size":        bs,
            })
            print(f"  Run {i+1}/{N_REPEATS}: "
                  f"tok/s={total_tok_per_sec:.1f}  "
                  f"per_prompt={per_prompt_ms:.0f}ms  "
                  f"tokens={total_new_tokens}")

        agg = {
            "batch_size":        bs,
            "total_tok_per_sec": round(statistics.mean(r["total_tok_per_sec"] for r in run_data), 2),
            "per_prompt_ms":     round(statistics.mean(r["per_prompt_ms"]     for r in run_data), 1),
            "wall_time_ms":      round(statistics.mean(r["wall_time_ms"]       for r in run_data), 1),
            "total_new_tokens":  run_data[-1]["total_new_tokens"],
            "engine":            "vllm",
        }
        results[bs] = agg
        print(f"  AVG: {agg['total_tok_per_sec']} tok/s  {agg['per_prompt_ms']}ms/prompt")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MANUAL BATCHING RESULTS FOR COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_manual_results_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p

    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "results" / "batching-results" / "batching_results.json",
        root / "results" / "batching_results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return p


def load_manual_results(path="batching_results.json"):
    resolved = _resolve_manual_results_path(path)
    if not resolved.exists():
        print(f"[WARN] {path} not found — skipping manual batching comparison.")
        print(f"       Also checked: {resolved}")
        print("       Run optimization/batching.py first, then re-run this script.")
        return {}
    with open(resolved) as f:
        data = json.load(f)
    results = {}
    for r in data:
        if r.get("status") == "oom_skipped":
            continue
        r["engine"] = "manual"
        results[r["batch_size"]] = r
    print(f"[LOAD] Loaded manual batching results from {resolved}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_comparison(vllm_res, manual_res, batch_sizes):
    print("\n")
    print("╔════════════╦════════════════════╦════════════════════╦═══════════════╗")
    print("║ Batch Size ║ Manual tok/sec     ║ vLLM tok/sec       ║ vLLM speedup  ║")
    print("╠════════════╬════════════════════╬════════════════════╬═══════════════╣")

    lines = []
    for bs in batch_sizes:
        v = vllm_res.get(bs)
        m = manual_res.get(bs)
        v_tps = v["total_tok_per_sec"] if v else None
        m_tps = m["total_tok_per_sec"] if m else None

        if v_tps and m_tps:
            speedup = f"{v_tps / m_tps:.2f}×"
        else:
            speedup = "—"

        m_str = f"{m_tps}" if m_tps else "—"
        v_str = f"{v_tps}" if v_tps else "—"
        print(f"║ {bs:<10} ║ {m_str:<18} ║ {v_str:<18} ║ {speedup:<13} ║")
        lines.append(f"| {bs} | {m_str} | {v_str} | {speedup} |")

    print("╚════════════╩════════════════════╩════════════════════╩═══════════════╝")

    # Save plain text version for README
    with open("vllm_comparison_table.txt", "w") as f:
        f.write("| Batch Size | Manual tok/sec | vLLM tok/sec | vLLM speedup |\n")
        f.write("|------------|---------------|--------------|---------------|\n")
        for line in lines:
            f.write(line + "\n")
    print("\n[SAVE] Plain text table → vllm_comparison_table.txt")


# ══════════════════════════════════════════════════════════════════════════════
#  HONEST OBSERVATIONS (printed after results)
# ══════════════════════════════════════════════════════════════════════════════
def print_observations(vllm_res, manual_res, batch_sizes):
    print("""
═══════════════════════════════════════════════════
  HONEST OBSERVATIONS: vLLM vs Manual Batching
═══════════════════════════════════════════════════

WHAT vLLM DOES DIFFERENTLY:
  • Continuous batching: new requests slot in as soon as
    any sequence in the current batch finishes — no waiting
    for the slowest sequence to complete.
  • PagedAttention: KV cache stored in non-contiguous
    memory pages (like virtual memory for GPU RAM).
    Eliminates memory fragmentation from variable-length seqs.
  • CUDA graph capture: first run traces the ops, subsequent
    runs replay the graph → lower kernel launch overhead.
  • Fused kernels: attention, softmax, etc. are custom CUDA
    kernels, not PyTorch's generic ones.

WHAT THE NUMBERS LIKELY SHOW:
  • For TinyLlama (1.1B) on T4 with batch=8 and only 100 tokens:
    vLLM may show modest improvement or similar numbers because:
    - The model is small enough that manual batching already
      saturates the GPU well (your 108% efficiency proves this)
    - Short generations (100 tokens) don't expose continuous
      batching's main advantage (long mixed-length sequences)
    - vLLM's overhead (engine startup, async scheduling) matters
      more at small scale

  • Where vLLM REALLY shines (not fully tested here):
    - Mixed-length requests (some 10 tokens, some 1000 tokens)
    - High concurrency (hundreds of simultaneous requests)
    - Long context (2048+ tokens where memory management matters)
    - Production servers where requests arrive continuously

BOTTOM LINE:
  Manual batching is fine for offline processing of uniform batches.
  vLLM is the correct choice for any real-time serving scenario.
  The architectural improvements matter at scale, not in toy benchmarks.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_comparison(vllm_res, manual_res, batch_sizes, path="vllm_comparison.png"):
    if not HAS_PLOT:
        return

    valid_bs = [bs for bs in batch_sizes if vllm_res.get(bs) or manual_res.get(bs)]
    if not valid_bs:
        return

    manual_tps = [manual_res.get(bs, {}).get("total_tok_per_sec", 0) for bs in valid_bs]
    vllm_tps   = [vllm_res.get(bs, {}).get("total_tok_per_sec",   0) for bs in valid_bs]

    x     = range(len(valid_bs))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f0f")

    # ── Left: throughput comparison ──
    ax1.set_facecolor("#1a1a2e")
    b1 = ax1.bar([i - width/2 for i in x], manual_tps, width,
                 label="Manual batching", color="#4ecdc4", alpha=0.9)
    b2 = ax1.bar([i + width/2 for i in x], vllm_tps,   width,
                 label="vLLM",            color="#ff6b6b", alpha=0.9)

    for bar, val in zip(list(b1) + list(b2), manual_tps + vllm_tps):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.0f}", ha="center", va="bottom",
                     color="white", fontsize=9)

    ax1.set_xlabel("Batch Size", color="white", fontsize=11)
    ax1.set_ylabel("Total Tokens per Second", color="white", fontsize=11)
    ax1.set_title("Throughput: Manual vs vLLM",
                  color="white", fontsize=13, fontweight="bold")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([str(bs) for bs in valid_bs])
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("#444")
    ax1.spines["left"].set_color("#444")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=10)

    # ── Right: per-prompt latency comparison ──
    manual_lat = [manual_res.get(bs, {}).get("per_prompt_ms", 0) for bs in valid_bs]
    vllm_lat   = [vllm_res.get(bs,   {}).get("per_prompt_ms", 0) for bs in valid_bs]

    ax2.set_facecolor("#1a1a2e")
    ax2.bar([i - width/2 for i in x], manual_lat, width,
            label="Manual batching", color="#4ecdc4", alpha=0.9)
    ax2.bar([i + width/2 for i in x], vllm_lat,   width,
            label="vLLM",            color="#ff6b6b", alpha=0.9)

    ax2.set_xlabel("Batch Size", color="white", fontsize=11)
    ax2.set_ylabel("Per-Prompt Latency (ms)", color="white", fontsize=11)
    ax2.set_title("Latency: Manual vs vLLM",
                  color="white", fontsize=13, fontweight="bold")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(bs) for bs in valid_bs])
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("#444")
    ax2.spines["left"].set_color("#444")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    print(f"[PLOT] Saved comparison chart → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--batch-sizes",  nargs="+", type=int, default=BATCH_SIZES)
    parser.add_argument("--max-tokens",   type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--compare-file", default="batching_results.json",
                        help="Path to manual batching_results.json for comparison")
    args = parser.parse_args()

    print_header(f"Phase 3 Step 3 — vLLM Dynamic Batching")
    print(f"  Model : {args.model}")

    # Compute maximum prompt length so we can set vLLM's max_tokens budget
    # (prompt + new tokens) and get meaningful generation lengths.
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_lens = [len(hf_tokenizer.encode(p, add_special_tokens=False)) for p in PROMPT_POOL]
    max_prompt_len = max(prompt_lens) if prompt_lens else 0
    max_model_len = max_prompt_len + args.max_tokens + 16

    # ── Load vLLM engine ──────────────────────────────────────────────────────
    print("\n[LOAD] Initialising vLLM engine (this takes ~60s on first run)...")
    print("       vLLM compiles CUDA kernels and captures computation graphs.")
    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=max_model_len,  # ensure enough room for prompt + new tokens
        gpu_memory_utilization=0.85,
    )
    print("[LOAD] vLLM engine ready.\n")

    # ── Run vLLM benchmark ────────────────────────────────────────────────────
    vllm_results  = run_vllm_benchmark(llm, args.batch_sizes, args.max_tokens, max_prompt_len=max_prompt_len)
    manual_results = load_manual_results(args.compare_file)

    # ── Results ───────────────────────────────────────────────────────────────
    print_header("COMPARISON RESULTS")
    print_comparison(vllm_results, manual_results, args.batch_sizes)
    print_observations(vllm_results, manual_results, args.batch_sizes)

    # ── Save + plot ───────────────────────────────────────────────────────────
    out = [r for r in vllm_results.values()]
    with open("vllm_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[SAVE] vLLM results → vllm_results.json")

    plot_comparison(vllm_results, manual_results, args.batch_sizes)

    print("\n✅ vLLM experiment complete!")
    print("   Files saved:")
    print("     vllm_results.json          ← raw vLLM metrics")
    print("     vllm_comparison.png        ← chart for README")
    print("     vllm_comparison_table.txt  ← markdown table")


if __name__ == "__main__":
    main()