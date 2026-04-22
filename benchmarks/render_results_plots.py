import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_kv_cache() -> None:
    data = load_json(RESULTS / "kv_cache_experiment-results" / "kv_cache_results.json")
    by_seq = {}
    for row in data:
        seq = int(row["seq_len"])
        by_seq.setdefault(seq, {})[row["label"]] = float(row["tok_per_sec"])

    seqs = sorted(by_seq.keys())
    speedups = []
    for s in seqs:
        on = by_seq[s].get("cache_on", 0.0)
        off = by_seq[s].get("cache_off", 1e-9)
        speedups.append(on / off if off > 0 else 0.0)

    out = RESULTS / "kv_cache_experiment-results" / "kv_cache_speedup.png"
    ensure_dir(out)
    plt.figure(figsize=(8, 4.5))
    plt.plot(seqs, speedups, marker="o")
    plt.title("KV Cache Speedup vs Context Length")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Speedup (cache_on / cache_off)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_vllm_comparison() -> None:
    vllm = load_json(RESULTS / "vllm" / "vllm_results.json")
    manual = load_json(RESULTS / "batching-results" / "batching_results.json")

    vllm_by_batch = {int(r["batch_size"]): float(r["total_tok_per_sec"]) for r in vllm}
    manual_by_batch = {int(r["batch_size"]): float(r["total_tok_per_sec"]) for r in manual}
    batches = sorted(set(vllm_by_batch.keys()) & set(manual_by_batch.keys()))

    out = RESULTS / "vllm" / "vllm_comparison.png"
    ensure_dir(out)
    plt.figure(figsize=(8, 4.5))
    plt.plot(batches, [vllm_by_batch[b] for b in batches], marker="o", label="vLLM")
    plt.plot(batches, [manual_by_batch[b] for b in batches], marker="o", label="Manual batching")
    plt.title("vLLM vs Manual Batching Throughput")
    plt.xlabel("Batch Size")
    plt.ylabel("Total Throughput (tok/s)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_load_test() -> None:
    summary = load_json(ROOT / "dashboard" / "public" / "data" / "load_test_summary.json")
    rows = summary.get("results", [])
    users = [int(r["users"]) for r in rows]
    rps = [float(r["generate_rps"]) for r in rows]
    p95 = [float(r["generate_p95_ms"]) for r in rows]

    out = RESULTS / "load-testing" / "load_test_concurrency.png"
    ensure_dir(out)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(users, rps, marker="o", color="#1f77b4")
    ax1.set_xlabel("Concurrent Users")
    ax1.set_ylabel("Generate RPS", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(users, p95, marker="o", color="#d62728")
    ax2.set_ylabel("P95 Latency (ms)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    plt.title("Load Test: Throughput and P95 Latency")
    fig.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_router_confusion() -> None:
    report = load_json(RESULTS / "router_eval_report.json")
    conf = report["confusion_matrix"]
    tiers = ["fast", "balanced", "quality"]
    matrix = [[int(conf[t][p]) for p in tiers] for t in tiers]

    out = RESULTS / "router_eval_confusion.png"
    ensure_dir(out)
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Router Confusion Matrix")
    plt.xticks(range(len(tiers)), tiers)
    plt.yticks(range(len(tiers)), tiers)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(len(tiers)):
        for j in range(len(tiers)):
            plt.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    plt.colorbar(shrink=0.85)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    plot_kv_cache()
    plot_vllm_comparison()
    plot_load_test()
    plot_router_confusion()
    print("Saved result plots for README.")


if __name__ == "__main__":
    main()
