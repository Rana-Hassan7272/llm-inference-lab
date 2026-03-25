"""
score_answers.py
────────────────
Interactive CLI tool to manually score answers from the JSON result files.

Run locally after downloading JSONs from Colab:
    python score_answers.py --file fp16_results.json

This walks through each question+answer, you type a score 1-5, and it saves
the updated JSON in place (with _scored suffix).

Scoring guide:
    5 — Perfectly correct, well-explained, complete
    4 — Correct with minor omissions or small errors
    3 — Partially correct, misses key points
    2 — Mostly wrong but shows some understanding
    1 — Completely wrong or incoherent
"""

import json
import argparse
import os
from pathlib import Path

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

SCORE_COLOURS = {1: RED, 2: RED, 3: YELLOW, 4: GREEN, 5: GREEN}

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def score_file(filepath):
    path = Path(filepath)
    if not path.exists():
        print(f"{RED}File not found: {filepath}{RESET}")
        return

    with open(path) as f:
        records = json.load(f)

    print(f"\n{BOLD}{CYAN}Scoring: {filepath}{RESET}")
    print(f"Questions: {len(records)}\n")
    print("Scoring guide: 5=Perfect  4=Good  3=Partial  2=Mostly wrong  1=Wrong/Incoherent\n")

    already_scored = sum(1 for r in records if r.get("quality_score") is not None)
    if already_scored > 0:
        print(f"{YELLOW}Note: {already_scored}/{len(records)} already scored. Re-scoring will overwrite.{RESET}")
        skip = input("Skip already-scored? [Y/n]: ").strip().lower()
        skip_scored = skip != "n"
    else:
        skip_scored = False

    for i, record in enumerate(records):
        if skip_scored and record.get("quality_score") is not None:
            continue

        clear()
        mode = record.get("mode", "?")
        print(f"{BOLD}{CYAN}[{i+1}/{len(records)}] Mode: {mode}{RESET}")
        print(f"{BOLD}Q: {record['question']}{RESET}\n")
        print(f"{DIM}Metrics: TTFT={record.get('ttft_ms')}ms | Tok/s={record.get('tok_per_sec')} | Mem={record.get('mem_gb')}GB{RESET}\n")
        print(f"{BOLD}Answer:{RESET}")
        print(f"{record['answer']}\n")
        print("─" * 60)

        while True:
            try:
                raw = input("Score (1-5, or 's' to skip, 'q' to quit & save): ").strip()
                if raw.lower() == "q":
                    # Save and exit
                    _save(records, path)
                    return
                if raw.lower() == "s":
                    break
                score = int(raw)
                if 1 <= score <= 5:
                    col = SCORE_COLOURS.get(score, RESET)
                    print(f"  → {col}{score}/5{RESET}")
                    record["quality_score"] = score
                    break
                else:
                    print(f"{RED}Enter a number between 1 and 5.{RESET}")
            except ValueError:
                print(f"{RED}Invalid input.{RESET}")

    _save(records, path)


def _save(records, original_path):
    out_path = original_path.with_name(original_path.stem + "_scored.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    scored_count = sum(1 for r in records if r.get("quality_score") is not None)
    print(f"\n{GREEN}Saved {scored_count}/{len(records)} scored answers → {out_path}{RESET}")

    # Also overwrite original so step4 script finds it
    with open(original_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"{GREEN}Also updated original → {original_path}{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually score LLM benchmark answers.")
    parser.add_argument("--file", nargs="+",
                        default=["fp16_results.json", "8bit_results.json", "4bit_results.json"],
                        help="JSON result file(s) to score")
    args = parser.parse_args()

    for f in args.file:
        print(f"\n{'='*60}")
        score_file(f)

    print(f"\n{BOLD}{GREEN}All files scored! Now run: python step4_comparison_table.py{RESET}\n")