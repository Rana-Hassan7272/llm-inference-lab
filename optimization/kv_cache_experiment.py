"""
optimization/kv_cache_experiment.py
=====================================
Phase 3 — Step 1: KV Cache Experiment

Measures generation speed with KV cache ENABLED vs DISABLED
at sequence lengths: 128, 256, 512, 1024 tokens.

WHERE TO RUN:
  → Recommended: Google Colab (free T4 GPU)
  → Also works:  Local machine with CUDA GPU
  → CPU fallback: Works but slow (~10-20 min). Use --seq-lens 128 256 only.

USAGE:
  # Colab / GPU (full experiment)
  python kv_cache_experiment.py

  # CPU only (shorter run)
  python kv_cache_experiment.py --cpu --seq-lens 128 256

  # Custom model
  python kv_cache_experiment.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

OUTPUTS:
  kv_cache_results.json      — raw numbers for Step 4 comparison table
  kv_cache_plot.png          — speedup graph (put this in your README)
  kv_cache_tokpersec.png     — tokens/sec side-by-side bar chart
"""

import torch
import time
import json
import argparse
import statistics
import sys
import os

# ── optional matplotlib ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # headless — works in Colab and servers
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib not found. Run: pip install matplotlib")

from transformers import AutoTokenizer, AutoModelForCausalLM

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_SEQ_LENS = [128, 256, 512, 1024]
DEFAULT_NEW_TOKENS = 64         # fixed generation length for fair comparisons
N_REPEATS        = 3          # average over N runs to reduce noise
WARMUP_RUNS      = 1          # discard first run (GPU warm-up)

# A long prompt that forces the model to generate up to target length.
# We truncate/pad the INPUT to a fixed size, then generate NEW tokens.
BASE_PROMPT = (
    "Explain in great detail the history, causes, major battles, political consequences, "
    "economic impacts, and long-term effects of the First World War. Cover the role of "
    "alliances, the assassination of Archduke Franz Ferdinand, trench warfare, the use "
    "of new weapons technology, the Treaty of Versailles, and how the war set the stage "
    "for the Second World War. Be thorough and comprehensive in your explanation."
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[WARN] No CUDA GPU found — running on CPU. This will be slow.")
    return torch.device("cpu")


def gpu_mem_gb():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_reserved() / 1024**3, 2)
    return 0.0


def print_header(text):
    bar = "═" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def print_row(label, value):
    print(f"  {label:<30} {value}")


# ══════════════════════════════════════════════════════════════════════════════
#  CORE BENCHMARK: single run at one sequence length, one cache setting
# ══════════════════════════════════════════════════════════════════════════════
def benchmark_single(
    model, tokenizer, input_ids, attention_mask, new_tokens, use_cache, device
):
    """
    Returns dict with:
      - tok_per_sec   : throughput during generation
      - ttft_ms       : time to first token (ms)
      - total_ms      : total generation wall time (ms)
      - new_tokens    : how many tokens were actually generated
      - mem_gb        : GPU memory at end
    """
    input_len = input_ids.shape[1]

    # ── Time to First Token ────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            min_new_tokens=1,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000

    # ── Full generation ────────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t1) * 1000

    new_tokens = output.shape[1] - input_len
    tok_per_sec = (new_tokens / (total_ms / 1000)) if total_ms > 0 else 0

    return {
        "tok_per_sec" : round(tok_per_sec, 2),
        "ttft_ms"     : round(ttft_ms, 1),
        "total_ms"    : round(total_ms, 1),
        "new_tokens"  : int(new_tokens),
        "mem_gb"      : gpu_mem_gb(),
    }


def build_context_tensors(base_ids, filler_ids, seq_len, device):
    """
    Build a single-example context with exact length = seq_len tokens.
    (This makes "sequence length" mean INPUT/context length, matching the roadmap.)
    """
    ids = base_ids[:seq_len].copy()
    while len(ids) < seq_len:
        ids.extend(filler_ids)
    ids = ids[:seq_len]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


# ══════════════════════════════════════════════════════════════════════════════
#  SWEEP: all sequence lengths × both cache settings
# ══════════════════════════════════════════════════════════════════════════════
def run_sweep(model, tokenizer, seq_lens, device, n_repeats=N_REPEATS):
    results = {}   # keyed by (seq_len, cache_mode)

    for seq_len in seq_lens:
        # Build a fixed-length context (INPUT tokens) for this seq_len.
        # We keep generation length fixed so differences come from KV-cache behavior.
        input_ids, attention_mask = build_context_tensors(
            base_ids=tokenizer._kv_cache_base_ids,   # injected in main() after tokenizer load
            filler_ids=tokenizer._kv_cache_filler_ids, # injected in main()
            seq_len=seq_len,
            device=device,
        )

        for use_cache, label in [(True, "cache_on"), (False, "cache_off")]:
            print_header(f"seq_len={seq_len}  |  cache={'ON' if use_cache else 'OFF'}")

            run_times = []

            # Warmup
            for _ in range(WARMUP_RUNS):
                benchmark_single(
                    model, tokenizer, input_ids, attention_mask,
                    new_tokens=tokenizer._kv_cache_new_tokens,
                    use_cache=use_cache, device=device
                )

            # Actual runs
            for i in range(n_repeats):
                r = benchmark_single(
                    model, tokenizer, input_ids, attention_mask,
                    new_tokens=tokenizer._kv_cache_new_tokens,
                    use_cache=use_cache, device=device
                )
                run_times.append(r)
                print_row(f"  Run {i+1}/{n_repeats}",
                          f"tok/s={r['tok_per_sec']:<8}  "
                          f"TTFT={r['ttft_ms']}ms  "
                          f"total={r['total_ms']}ms  "
                          f"tokens={r['new_tokens']}")

            # Aggregate
            agg = {
                "seq_len"     : seq_len,
                "use_cache"   : use_cache,
                "label"       : label,
                "tok_per_sec" : round(statistics.mean(r["tok_per_sec"] for r in run_times), 2),
                "ttft_ms"     : round(statistics.mean(r["ttft_ms"]     for r in run_times), 1),
                "total_ms"    : round(statistics.mean(r["total_ms"]    for r in run_times), 1),
                "new_tokens"  : run_times[-1]["new_tokens"],
                "mem_gb"      : run_times[-1]["mem_gb"],
                "raw_runs"    : run_times,
            }
            results[(seq_len, label)] = agg

            print_row("  AVG tok/s :", agg["tok_per_sec"])
            print_row("  AVG TTFT  :", f"{agg['ttft_ms']} ms")
            print_row("  GPU mem   :", f"{agg['mem_gb']} GB")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Speedup ratio (cache_on / cache_off) per sequence length
# ══════════════════════════════════════════════════════════════════════════════
def plot_speedup(results, seq_lens, output_path="kv_cache_speedup.png"):
    if not HAS_PLOT:
        return

    speedups = []
    for sl in seq_lens:
        on  = results[(sl, "cache_on")]["tok_per_sec"]
        off = results[(sl, "cache_off")]["tok_per_sec"]
        speedups.append(on / off if off > 0 else 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a2e")

    bars = ax.bar(
        [str(s) for s in seq_lens], speedups,
        color=["#00d4ff" if s >= 1 else "#ff4757" for s in speedups],
        edgecolor="#ffffff22", linewidth=0.8, width=0.55
    )

    # Value labels on bars
    for bar, val in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}×",
            ha="center", va="bottom",
            color="white", fontsize=12, fontweight="bold"
        )

    ax.axhline(y=1.0, color="#ff6b6b", linestyle="--", linewidth=1.5, alpha=0.8,
               label="No speedup (1×)")

    ax.set_xlabel("Context Length (input tokens)", color="white", fontsize=12)
    ax.set_ylabel("Speedup (Cache ON / Cache OFF)", color="white", fontsize=12)
    ax.set_title("KV Cache Speedup vs Sequence Length\n"
                 "Longer sequences → More dramatic cache benefit",
                 color="white", fontsize=14, fontweight="bold", pad=15)

    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=10)
    ax.set_ylim(0, max(speedups) * 1.25 + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0f0f0f")
    plt.close()
    print(f"\n[PLOT] Saved speedup chart → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Tokens/sec side-by-side (cache on vs off) at each seq length
# ══════════════════════════════════════════════════════════════════════════════
def plot_tokpersec(results, seq_lens, output_path="kv_cache_tokpersec.png"):
    if not HAS_PLOT:
        return

    on_vals  = [results[(sl, "cache_on")]["tok_per_sec"]  for sl in seq_lens]
    off_vals = [results[(sl, "cache_off")]["tok_per_sec"] for sl in seq_lens]

    x     = range(len(seq_lens))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a2e")

    bars_on  = ax.bar([i - width/2 for i in x], on_vals,  width,
                      label="Cache ON",  color="#00d4ff", alpha=0.9)
    bars_off = ax.bar([i + width/2 for i in x], off_vals, width,
                      label="Cache OFF", color="#ff6b6b", alpha=0.9)

    for bar, val in zip(list(bars_on) + list(bars_off), on_vals + off_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}",
            ha="center", va="bottom",
            color="white", fontsize=9
        )

    ax.set_xlabel("Context Length (input tokens)", color="white", fontsize=12)
    ax.set_ylabel("Tokens per Second", color="white", fontsize=12)
    ax.set_title("Throughput: KV Cache ON vs OFF",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0f0f0f")
    plt.close()
    print(f"[PLOT] Saved tok/sec chart → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — TTFT comparison (latency of first token)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ttft(results, seq_lens, output_path="kv_cache_ttft.png"):
    if not HAS_PLOT:
        return

    on_vals  = [results[(sl, "cache_on")]["ttft_ms"]  for sl in seq_lens]
    off_vals = [results[(sl, "cache_off")]["ttft_ms"] for sl in seq_lens]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a2e")

    ax.plot([str(s) for s in seq_lens], on_vals,
            "o-", color="#00d4ff", linewidth=2.5, markersize=8, label="Cache ON")
    ax.plot([str(s) for s in seq_lens], off_vals,
            "s--", color="#ff6b6b", linewidth=2.5, markersize=8, label="Cache OFF")

    for i, (s, on, off) in enumerate(zip(seq_lens, on_vals, off_vals)):
        ax.annotate(f"{on:.0f}ms", (str(s), on),
                    textcoords="offset points", xytext=(0, 10),
                    color="#00d4ff", fontsize=9, ha="center")
        ax.annotate(f"{off:.0f}ms", (str(s), off),
                    textcoords="offset points", xytext=(0, -18),
                    color="#ff6b6b", fontsize=9, ha="center")

    ax.set_xlabel("Context Length (input tokens)", color="white", fontsize=12)
    ax.set_ylabel("Time to First Token (ms)", color="white", fontsize=12)
    ax.set_title("Time to First Token: KV Cache ON vs OFF",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0f0f0f")
    plt.close()
    print(f"[PLOT] Saved TTFT chart → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(results, seq_lens):
    print("\n")
    print("╔══════════╦═══════════════╦═══════════════╦══════════╦═══════════╗")
    print("║ Seq Len  ║ Cache ON      ║ Cache OFF     ║ Speedup  ║ TTFT diff ║")
    print("║ (context)║ (tok/sec)     ║ (tok/sec)     ║          ║           ║")
    print("╠══════════╬═══════════════╬═══════════════╬══════════╬═══════════╣")
    for sl in seq_lens:
        on     = results[(sl, "cache_on")]
        off    = results[(sl, "cache_off")]
        spd    = on["tok_per_sec"] / off["tok_per_sec"] if off["tok_per_sec"] > 0 else 0
        tdiff  = off["ttft_ms"] - on["ttft_ms"]
        print(f"║ {sl:<8} ║ {on['tok_per_sec']:<13} ║ {off['tok_per_sec']:<13} ║ {spd:.2f}×    ║ {tdiff:+.0f}ms    ║")
    print("╚══════════╩═══════════════╩═══════════════╩══════════╩═══════════╝")


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS JSON
# ══════════════════════════════════════════════════════════════════════════════
def save_results(results, seq_lens, path="kv_cache_results.json"):
    out = []
    for sl in seq_lens:
        for label in ["cache_on", "cache_off"]:
            r = results[(sl, label)].copy()
            r.pop("raw_runs", None)   # don't bloat the JSON
            out.append(r)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] Results saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="KV Cache Benchmark")
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help="HuggingFace model ID")
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=DEFAULT_SEQ_LENS,
                        help="Sequence lengths to test, e.g. --seq-lens 128 256 512")
    parser.add_argument("--new-tokens", type=int, default=DEFAULT_NEW_TOKENS,
                        help="Fixed number of new tokens to generate per run (default: 64)")
    parser.add_argument("--repeats",  type=int, default=N_REPEATS,
                        help="Number of timed runs per config (default: 3)")
    parser.add_argument("--cpu",      action="store_true",
                        help="Force CPU (slow — use short seq-lens only)")
    args = parser.parse_args()

    device = get_device(force_cpu=args.cpu)

    print_header(f"KV Cache Experiment  |  device={device}")
    print_row("Model      :", args.model)
    print_row("Seq lengths:", str(args.seq_lens))
    print_row("Repeats    :", args.repeats)
    print_row("Device     :", str(device))
    if device.type == "cuda":
        print_row("GPU        :", torch.cuda.get_device_name(0))
        print_row("VRAM total :", f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n[LOAD] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Precompute fixed token sequences so "seq_len" means INPUT/context length.
    base_ids = tokenizer.encode(BASE_PROMPT, add_special_tokens=False)
    filler_ids = tokenizer.encode(" The quick brown fox jumps over the lazy dog.", add_special_tokens=False)
    if not filler_ids:
        filler_ids = [tokenizer.eos_token_id]

    # Inject into tokenizer so run_sweep can build contexts without expanding function signatures.
    tokenizer._kv_cache_base_ids = base_ids
    tokenizer._kv_cache_filler_ids = filler_ids
    tokenizer._kv_cache_new_tokens = args.new_tokens

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
    results = run_sweep(model, tokenizer, args.seq_lens, device, n_repeats=args.repeats)

    # ── Summary ────────────────────────────────────────────────────────────────
    print_header("RESULTS SUMMARY")
    print_summary(results, args.seq_lens)

    # ── Key insight ────────────────────────────────────────────────────────────
    print("\n[INSIGHT] KV Cache benefit grows with sequence length:")
    for sl in args.seq_lens:
        on  = results[(sl, "cache_on")]["tok_per_sec"]
        off = results[(sl, "cache_off")]["tok_per_sec"]
        spd = on / off if off > 0 else 0
        bar = "█" * int(spd * 10)
        print(f"  seq={sl:<5}  {spd:.2f}x  {bar}")

    print("""
[WHY] Without KV cache, every new token must recompute attention
      over ALL previous tokens → O(n²) compute per token.
      With KV cache, only the NEW token needs attention computed
      against STORED keys/values → O(n) per token.
      This is why longer sequences show bigger speedups.
    """)

    # ── Save & plot ────────────────────────────────────────────────────────────
    save_results(results, args.seq_lens)
    plot_speedup(results,   args.seq_lens, "kv_cache_speedup.png")
    plot_tokpersec(results, args.seq_lens, "kv_cache_tokpersec.png")
    plot_ttft(results,      args.seq_lens, "kv_cache_ttft.png")

    print("\n✅ KV Cache experiment complete!")
    print("   Files saved:")
    print("     kv_cache_results.json   ← use in Step 4 comparison table")
    print("     kv_cache_speedup.png    ← put this in your README ⭐")
    print("     kv_cache_tokpersec.png  ← tok/sec bar chart")
    print("     kv_cache_ttft.png       ← latency line chart")


if __name__ == "__main__":
    main()