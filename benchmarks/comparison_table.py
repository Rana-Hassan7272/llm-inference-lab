"""
comparison_table.py
───────────────────
Build a quantization comparison table from benchmark JSON files.

Usage:
    python benchmarks/comparison_table.py

Each file is a list of dicts with keys:
    mode, question_id, question, answer,
    ttft_ms, tok_per_sec, mem_gb, quality_score

Set quality_score (1-5) manually in each JSON before running this script.
"""

import json
import argparse
import statistics
from pathlib import Path


# ── ANSI colours (works on Linux/Mac/Windows Terminal) ────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def resolve_path(filename):
    # Accept either explicit path or basename living in ../results/
    p = Path(filename)
    if p.exists():
        return p
    return Path(__file__).resolve().parent.parent / "results" / filename


def load(filename):
    path = resolve_path(filename)
    if not path.exists():
        print(f"{RED}[ERROR] File not found: {filename} (looked for {path}){RESET}")
        return []
    with open(path) as f:
        return json.load(f)


def aggregate(records):
    """Compute mean metrics across all 20 questions."""
    if not records:
        return None

    mem_values   = [r["mem_gb"]      for r in records if r["mem_gb"]      is not None]
    ttft_values  = [r["ttft_ms"]     for r in records if r["ttft_ms"]     is not None]
    toks_values  = [r["tok_per_sec"] for r in records if r["tok_per_sec"] is not None]
    qual_values  = [r["quality_score"] for r in records if r["quality_score"] is not None]

    return {
        "mode"       : records[0]["mode"],
        "mem_gb"     : round(statistics.mean(mem_values),  2) if mem_values  else "N/A",
        "ttft_ms"    : round(statistics.mean(ttft_values), 1) if ttft_values else "N/A",
        "tok_per_sec": round(statistics.mean(toks_values), 1) if toks_values else "N/A",
        "quality"    : round(statistics.mean(qual_values), 2) if qual_values else "Not scored",
        "n_scored"   : len(qual_values),
    }


def colour_best(values, idx, lower_is_better=False):
    """Return ANSI colour: green = best, yellow = middle, red = worst."""
    numeric = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if len(numeric) < 2:
        return RESET
    sorted_vals = sorted(numeric, key=lambda x: x[1], reverse=not lower_is_better)
    rank = {i: r for r, (i, _) in enumerate(sorted_vals)}
    r = rank.get(idx, 1)
    if r == 0:   return GREEN
    if r == len(sorted_vals) - 1: return RED
    return YELLOW


def print_table(rows):
    """Pretty-print the comparison table."""
    COL_W = [10, 10, 10, 10, 14]
    headers = ["Mode", "Memory", "Latency", "Tok/sec", "Quality"]

    sep  = "─" * (sum(COL_W) + len(COL_W) * 3 + 1)
    hsep = "═" * (sum(COL_W) + len(COL_W) * 3 + 1)

    def row_str(cells, colours=None):
        parts = []
        for i, (cell, w) in enumerate(zip(cells, COL_W)):
            col = colours[i] if colours else RESET
            parts.append(f"{col}{str(cell):<{w}}{RESET}")
        return "│ " + " │ ".join(parts) + " │"

    print(f"\n{BOLD}{CYAN}{hsep}{RESET}")
    print(f"{BOLD}{CYAN}  QUANTIZATION COMPARISON RESULTS{RESET}")
    print(f"{BOLD}{CYAN}{hsep}{RESET}")

    # Header
    print(row_str(headers, [BOLD]*5))
    print(sep)

    # Data rows
    mem_vals  = [r["mem_gb"]      for r in rows]
    ttft_vals = [r["ttft_ms"]     for r in rows]
    toks_vals = [r["tok_per_sec"] for r in rows]
    qual_vals = [r["quality"] if isinstance(r["quality"], float) else -1 for r in rows]

    for i, r in enumerate(rows):
        qual_display = (
            f"{r['quality']}/5 ({r['n_scored']}/20)"
            if isinstance(r["quality"], float)
            else r["quality"]
        )
        cells = [
            r["mode"],
            f"{r['mem_gb']} GB"      if isinstance(r["mem_gb"],      float) else r["mem_gb"],
            f"{r['ttft_ms']} ms"     if isinstance(r["ttft_ms"],     float) else r["ttft_ms"],
            f"{r['tok_per_sec']}"    if isinstance(r["tok_per_sec"], float) else r["tok_per_sec"],
            qual_display,
        ]
        colours = [
            RESET,
            colour_best(mem_vals,  i, lower_is_better=True),
            colour_best(ttft_vals, i, lower_is_better=True),
            colour_best(toks_vals, i, lower_is_better=False),
            colour_best(qual_vals, i, lower_is_better=False),
        ]
        print(row_str(cells, colours))

    print(f"{hsep}\n")


def print_per_question_detail(all_records_by_mode):
    """Optional: print per-question breakdown."""
    print(f"\n{BOLD}{'─'*80}{RESET}")
    print(f"{BOLD}  PER-QUESTION BREAKDOWN{RESET}")
    print(f"{'─'*80}")

    modes = list(all_records_by_mode.keys())
    n = max(len(v) for v in all_records_by_mode.values())

    for q_idx in range(n):
        # Get question text from first available mode
        q_text = None
        for m in modes:
            recs = all_records_by_mode[m]
            if q_idx < len(recs):
                q_text = recs[q_idx]["question"]
                break

        print(f"\n{CYAN}Q{q_idx+1}: {q_text}{RESET}")
        for mode in modes:
            recs = all_records_by_mode[mode]
            if q_idx < len(recs):
                r = recs[q_idx]
                qs = r["quality_score"]
                qs_str = f"{qs}/5" if qs is not None else f"{RED}not scored{RESET}"
                print(f"  {BOLD}{mode:<8}{RESET} | Tok/s={r['tok_per_sec']:<6} | "
                      f"TTFT={r['ttft_ms']:<7}ms | Quality={qs_str}")
                print(f"           {DIM}Answer: {str(r['answer'])[:120]}{RESET}")


def export_csv(rows, path="comparison_results.csv"):
    """Write a CSV to disk."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode","mem_gb","ttft_ms","tok_per_sec","quality","n_scored"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FP16/8-bit/4-bit benchmark results.")
    parser.add_argument(
        "--files",
        nargs=3,
        metavar=("FP16", "BIT8", "BIT4"),
        default=["fp16_results.json", "8bit_results.json", "4bit_results.json"],
        help="Input files. Basenames are auto-resolved under results/.",
    )
    parser.add_argument(
        "--csv-out",
        default="comparison_results.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    print(f"{BOLD}Loading result files...{RESET}")

    fp16_records  = load(args.files[0])
    bit8_records  = load(args.files[1])
    bit4_records  = load(args.files[2])

    rows = []
    all_records = {}

    for label, records in [("FP16", fp16_records), ("8-bit", bit8_records), ("4-bit", bit4_records)]:
        if records:
            agg = aggregate(records)
            rows.append(agg)
            all_records[label] = records
            print(f"  {GREEN}✓{RESET} {label}: {len(records)} questions loaded")
        else:
            print(f"  {RED}✗{RESET} {label}: skipped (no data)")

    if not rows:
        print(f"\n{RED}No results found. Run the Colab notebook first.{RESET}")
        exit(1)

    # ── Main comparison table ──
    print_table(rows)

    # ── Observations summary ──
    print(f"{BOLD}Observations:{RESET}")
    if len(rows) >= 2:
        fp16 = next((r for r in rows if r["mode"] == "FP16"), None)
        b8   = next((r for r in rows if r["mode"] == "8-bit"), None)
        b4   = next((r for r in rows if r["mode"] == "4-bit"), None)

        if fp16 and b8:
            mem_saving = round(fp16["mem_gb"] - b8["mem_gb"], 2) if isinstance(fp16["mem_gb"], float) and isinstance(b8["mem_gb"], float) else "?"
            print(f"  • FP16 → 8-bit: saves ~{mem_saving} GB memory")
        if fp16 and b4:
            mem_saving = round(fp16["mem_gb"] - b4["mem_gb"], 2) if isinstance(fp16["mem_gb"], float) and isinstance(b4["mem_gb"], float) else "?"
            print(f"  • FP16 → 4-bit: saves ~{mem_saving} GB memory")
        if b8 and isinstance(b8["quality"], float) and fp16 and isinstance(fp16["quality"], float):
            drop = round(fp16["quality"] - b8["quality"], 2)
            print(f"  • Quality drop FP16 → 8-bit: {drop}/5 points")
        if b4 and isinstance(b4["quality"], float) and fp16 and isinstance(fp16["quality"], float):
            drop = round(fp16["quality"] - b4["quality"], 2)
            print(f"  • Quality drop FP16 → 4-bit: {drop}/5 points")

    print()

    # ── Optional detailed breakdown ──
    show_detail = input("Show per-question breakdown? [y/N]: ").strip().lower()
    if show_detail == "y":
        print_per_question_detail(all_records)

    # ── Export ──
    export_csv(rows, args.csv_out)

    print(f"\n{GREEN}{BOLD}Done!{RESET} Check {args.csv_out}\n")