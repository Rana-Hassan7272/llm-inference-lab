"""
optimization/batching.py
=========================
Phase 3 — Step 2: Static Batching Experiment

Processes multiple prompts simultaneously and measures throughput
at batch sizes: 1, 2, 4, 8.

Records:
  - Tokens per second (total across whole batch)
  - Latency per prompt (ms per individual response)
  - GPU memory used
  - Batch efficiency (how close to linear scaling)

WHERE TO RUN:
  → Recommended: Google Colab (free T4 GPU)
  → Local GPU: Works great
  → CPU: Works but very slow. Use --batch-sizes 1 2 only.

USAGE:
  # Colab / GPU (full experiment)
  python batching.py

  # CPU only
  python batching.py --cpu --batch-sizes 1 2

  # Custom model or batch sizes
  python batching.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --batch-sizes 1 2 4 8

OUTPUTS:
  batching_results.json       — raw data for Step 4 comparison
  batching_throughput.png     — total tok/sec vs batch size
  batching_latency.png        — per-prompt latency vs batch size
  batching_efficiency.png     — scaling efficiency chart
"""

import torch
import time
import json
import argparse
import statistics
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib not found. Run: pip install matplotlib")

from transformers import AutoTokenizer, AutoModelForCausalLM

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_BATCH_SIZES = [1, 2, 4, 8]
MAX_NEW_TOKENS      = 100      # keep short so batching fits in VRAM
N_REPEATS           = 3
WARMUP_RUNS         = 1

# Diverse prompts — used to fill batches. We cycle through them.
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


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[WARN] No CUDA GPU found — running on CPU.")
    return torch.device("cpu")


def gpu_mem_gb():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_reserved() / 1024**3, 2)
    return 0.0


def print_header(text):
    bar = "═" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def get_batch_prompts(batch_size):
    """Return a list of `batch_size` prompts, cycling through the pool."""
    return [PROMPT_POOL[i % len(PROMPT_POOL)] for i in range(batch_size)]


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE BATCH RUN
# ══════════════════════════════════════════════════════════════════════════════
def run_batch(model, tokenizer, prompts, max_new_tokens, device):
    """
    Tokenize `prompts` as a padded batch, generate, return metrics.

    Returns:
      total_tok_per_sec   : tokens across entire batch / wall time
      per_prompt_ms       : wall time / num prompts (latency per request)
      total_new_tokens    : sum of generated tokens across all prompts
      mem_gb              : GPU memory after generation
    """
    batch_size = len(prompts)

    # ── Tokenize with left-padding (required for batched generation) ───────────
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,   # cap input length so batch fits in VRAM
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,     # always use cache for batching — we're measuring batch size effect
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0

    # Count generated tokens (subtract input padding)
    total_new_tokens = 0
    for i in range(batch_size):
        # actual new tokens = output length - input length (before padding)
        gen_tokens = outputs[i].shape[0] - input_len
        total_new_tokens += max(gen_tokens, 0)

    total_tok_per_sec = total_new_tokens / wall_time if wall_time > 0 else 0
    per_prompt_ms     = (wall_time / batch_size) * 1000

    return {
        "total_tok_per_sec"  : round(total_tok_per_sec, 2),
        "per_prompt_ms"      : round(per_prompt_ms, 1),
        "total_new_tokens"   : int(total_new_tokens),
        "wall_time_ms"       : round(wall_time * 1000, 1),
        "mem_gb"             : gpu_mem_gb(),
        "batch_size"         : batch_size,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SWEEP: all batch sizes
# ══════════════════════════════════════════════════════════════════════════════
def run_sweep(model, tokenizer, batch_sizes, device, max_new_tokens=MAX_NEW_TOKENS):
    results = {}

    for bs in batch_sizes:
        print_header(f"Batch Size = {bs}")
        prompts = get_batch_prompts(bs)

        # Warmup
        for _ in range(WARMUP_RUNS):
            try:
                run_batch(model, tokenizer, prompts, max_new_tokens, device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM] Batch size {bs} too large for GPU VRAM — skipping.")
                    results[bs] = None
                    break
                raise

        if bs in results and results[bs] is None:
            continue

        run_data = []
        oom = False
        for i in range(N_REPEATS):
            try:
                r = run_batch(model, tokenizer, prompts, max_new_tokens, device)
                run_data.append(r)
                print(f"  Run {i+1}/{N_REPEATS}: "
                      f"total={r['total_tok_per_sec']:.1f} tok/s  "
                      f"per_prompt={r['per_prompt_ms']:.0f}ms  "
                      f"mem={r['mem_gb']}GB")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM] Out of memory at batch size {bs} — skipping rest.")
                    oom = True
                    break
                raise

        if oom or not run_data:
            results[bs] = None
            continue

        agg = {
            "batch_size"         : bs,
            "total_tok_per_sec"  : round(statistics.mean(r["total_tok_per_sec"] for r in run_data), 2),
            "per_prompt_ms"      : round(statistics.mean(r["per_prompt_ms"]      for r in run_data), 1),
            "total_new_tokens"   : run_data[-1]["total_new_tokens"],
            "wall_time_ms"       : round(statistics.mean(r["wall_time_ms"]        for r in run_data), 1),
            "mem_gb"             : run_data[-1]["mem_gb"],
            "raw_runs"           : run_data,
        }
        results[bs] = agg

        print(f"  ─── AVG: {agg['total_tok_per_sec']} tok/s  "
              f"{agg['per_prompt_ms']}ms/prompt  "
              f"mem={agg['mem_gb']}GB")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(results, batch_sizes):
    print("\n")
    # baseline = batch size 1
    bs1 = results.get(1)
    baseline_tps = bs1["total_tok_per_sec"] if bs1 else None

    print("╔════════════╦══════════════════╦════════════════╦═══════════╦══════════════╗")
    print("║ Batch Size ║ Total Tok/sec    ║ Per-prompt ms  ║ GPU Mem   ║ Efficiency   ║")
    print("╠════════════╬══════════════════╬════════════════╬═══════════╬══════════════╣")
    for bs in batch_sizes:
        r = results.get(bs)
        if r is None:
            print(f"║ {bs:<10} ║ {'OOM / SKIPPED':<16} ║ {'—':<14} ║ {'—':<9} ║ {'—':<12} ║")
            continue
        # Efficiency: actual speedup vs ideal linear scaling
        if baseline_tps and baseline_tps > 0:
            actual_speedup = r["total_tok_per_sec"] / baseline_tps
            ideal_speedup  = bs / 1
            efficiency     = (actual_speedup / ideal_speedup) * 100
            eff_str        = f"{efficiency:.0f}%"
        else:
            eff_str = "—"
        print(f"║ {bs:<10} ║ {r['total_tok_per_sec']:<16} ║ {r['per_prompt_ms']:<14} ║ {r['mem_gb']:<9} ║ {eff_str:<12} ║")
    print("╚════════════╩══════════════════╩════════════════╩═══════════╩══════════════╝")

    print("""
EFFICIENCY EXPLAINED:
  100% = perfect linear scaling (doubling batch → doubles throughput)
  GPU compute is the bottleneck below 100%. Memory bandwidth at 100%+.
  Typically you'll see 60-85% efficiency — still a major win vs sequential.
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Total throughput vs batch size
# ══════════════════════════════════════════════════════════════════════════════
def plot_throughput(results, batch_sizes, output_path="batching_throughput.png"):
    if not HAS_PLOT:
        return

    valid = [(bs, results[bs]) for bs in batch_sizes if results.get(bs)]
    if not valid:
        return

    xs    = [bs for bs, _ in valid]
    tps   = [r["total_tok_per_sec"] for _, r in valid]
    b1    = tps[0] if tps else 1
    ideal = [b1 * (bs / xs[0]) for bs in xs]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a2e")

    ax.plot(xs, ideal, "--", color="#ffffff44", linewidth=1.5, label="Ideal linear scaling")
    ax.plot(xs, tps, "o-", color="#00d4ff", linewidth=2.5, markersize=10, label="Actual throughput")

    for x, y in zip(xs, tps):
        ax.annotate(f"{y:.1f}", (x, y),
                    textcoords="offset points", xytext=(0, 12),
                    color="white", fontsize=10, ha="center", fontweight="bold")

    ax.fill_between(xs, ideal, tps, alpha=0.12,
                    color="#ff6b6b", label="Gap from ideal")

    ax.set_xlabel("Batch Size (prompts processed simultaneously)", color="white", fontsize=12)
    ax.set_ylabel("Total Tokens per Second", color="white", fontsize=12)
    ax.set_title("Batching Throughput vs Batch Size\n"
                 "Dashed line = perfect linear scaling",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(xs)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    print(f"[PLOT] Saved throughput chart → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Per-prompt latency vs batch size
# ══════════════════════════════════════════════════════════════════════════════
def plot_latency(results, batch_sizes, output_path="batching_latency.png"):
    if not HAS_PLOT:
        return

    valid = [(bs, results[bs]) for bs in batch_sizes if results.get(bs)]
    if not valid:
        return

    xs  = [bs for bs, _ in valid]
    lat = [r["per_prompt_ms"] for _, r in valid]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a2e")

    bars = ax.bar([str(x) for x in xs], lat,
                  color=["#00d4ff", "#00b4d8", "#0096c7", "#0077b6"][:len(xs)],
                  edgecolor="#ffffff22", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, lat):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{val:.0f}ms",
                ha="center", va="bottom",
                color="white", fontsize=11, fontweight="bold")

    ax.set_xlabel("Batch Size", color="white", fontsize=12)
    ax.set_ylabel("Per-Prompt Latency (ms)", color="white", fontsize=12)
    ax.set_title("Per-Prompt Latency vs Batch Size\n"
                 "Latency rises with batch size but throughput improves",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    print(f"[PLOT] Saved latency chart → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — Batch efficiency (actual vs ideal %)
# ══════════════════════════════════════════════════════════════════════════════
def plot_efficiency(results, batch_sizes, output_path="batching_efficiency.png"):
    if not HAS_PLOT:
        return

    valid = [(bs, results[bs]) for bs in batch_sizes if results.get(bs)]
    if len(valid) < 2:
        return

    b1_tps = valid[0][1]["total_tok_per_sec"]
    xs     = [bs for bs, _ in valid]
    effs   = []
    for bs, r in valid:
        actual  = r["total_tok_per_sec"] / b1_tps
        ideal   = bs / valid[0][0]
        effs.append((actual / ideal) * 100 if ideal > 0 else 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a2e")

    colours = ["#00d4ff" if e >= 80 else "#ffd166" if e >= 60 else "#ff6b6b" for e in effs]
    bars = ax.bar([str(x) for x in xs], effs,
                  color=colours, edgecolor="#ffffff22", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, effs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}%",
                ha="center", va="bottom",
                color="white", fontsize=12, fontweight="bold")

    ax.axhline(y=100, color="#ffffff44", linestyle="--", linewidth=1.5)
    ax.axhline(y=80,  color="#ffd16644", linestyle=":",  linewidth=1.2)
    ax.text(len(xs) - 0.5, 101, "Ideal (100%)", color="#ffffff88", fontsize=9)
    ax.text(len(xs) - 0.5, 81,  "Good (80%)",   color="#ffd16688", fontsize=9)

    ax.set_xlabel("Batch Size", color="white", fontsize=12)
    ax.set_ylabel("Scaling Efficiency (%)", color="white", fontsize=12)
    ax.set_title("Batch Scaling Efficiency\n"
                 "100% = perfectly linear scaling (rarely achieved)",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 120)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    print(f"[PLOT] Saved efficiency chart → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS JSON
# ══════════════════════════════════════════════════════════════════════════════
def save_results(results, batch_sizes, path="batching_results.json"):
    out = []
    for bs in batch_sizes:
        r = results.get(bs)
        if r is None:
            out.append({"batch_size": bs, "status": "oom_skipped"})
        else:
            row = r.copy()
            row.pop("raw_runs", None)
            out.append(row)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] Results saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Static Batching Benchmark")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=DEFAULT_BATCH_SIZES,
                        help="Batch sizes to test, e.g. --batch-sizes 1 2 4 8")
    parser.add_argument("--max-tokens",  type=int, default=MAX_NEW_TOKENS,
                        help="Max new tokens per prompt (default: 100)")
    parser.add_argument("--repeats",     type=int, default=N_REPEATS)
    parser.add_argument("--cpu",         action="store_true")
    args = parser.parse_args()

    device = get_device(force_cpu=args.cpu)

    print_header(f"Static Batching Experiment  |  device={device}")
    print(f"  Model       : {args.model}")
    print(f"  Batch sizes : {args.batch_sizes}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Repeats     : {args.repeats}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total  : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n[LOAD] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()

    print(f"[LOAD] Model loaded. GPU mem: {gpu_mem_gb()} GB")

    # ── Run sweep ──────────────────────────────────────────────────────────────
    results = run_sweep(model, tokenizer, args.batch_sizes, device,
                        max_new_tokens=args.max_tokens)

    # ── Summary ────────────────────────────────────────────────────────────────
    print_header("RESULTS SUMMARY")
    print_summary(results, args.batch_sizes)

    # ── Key insight bar chart ──────────────────────────────────────────────────
    print("\n[INSIGHT] Throughput scaling:")
    b1 = results.get(1)
    b1_tps = b1["total_tok_per_sec"] if b1 else 1
    for bs in args.batch_sizes:
        r = results.get(bs)
        if r:
            mult = r["total_tok_per_sec"] / b1_tps
            bar  = "█" * int(mult * 8)
            print(f"  batch={bs:<3}  {r['total_tok_per_sec']:.1f} tok/s  ({mult:.2f}× vs batch=1)  {bar}")

    print("""
[WHY] Static batching works because:
  • GPU has thousands of CUDA cores that can process in parallel
  • A single prompt uses a fraction of available compute
  • Batching fills the GPU properly → better utilization
  • Trade-off: higher latency per prompt, but far better throughput
  • Real inference servers (vLLM, TGI) use dynamic batching
    to get the best of both worlds
    """)

    # ── Save & plot ────────────────────────────────────────────────────────────
    save_results(results, args.batch_sizes)
    plot_throughput(results, args.batch_sizes, "batching_throughput.png")
    plot_latency(results,    args.batch_sizes, "batching_latency.png")
    plot_efficiency(results, args.batch_sizes, "batching_efficiency.png")

    print("\n✅ Batching experiment complete!")
    print("   Files saved:")
    print("     batching_results.json       ← use in Step 4 comparison table")
    print("     batching_throughput.png     ← put this in your README ⭐")
    print("     batching_latency.png        ← per-prompt latency chart")
    print("     batching_efficiency.png     ← scaling efficiency chart")


if __name__ == "__main__":
    main()