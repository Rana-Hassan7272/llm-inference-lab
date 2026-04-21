import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "benchmarks" / "router_eval_dataset.json"
DEFAULT_OUT = ROOT / "results" / "router_eval_report.json"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TIERS = ["fast", "balanced", "quality"]

# Ensure project-root imports (e.g., inference.adaptive_router) work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.adaptive_router import route


def load_dataset(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = []
    for row in data:
        prompt = (row.get("prompt") or "").strip()
        expected = (row.get("expected_tier") or "").strip().lower()
        if prompt and expected in TIERS:
            cleaned.append({"prompt": prompt, "expected_tier": expected})
    return cleaned


def empty_confusion() -> Dict[str, Dict[str, int]]:
    return {exp: {pred: 0 for pred in TIERS} for exp in TIERS}


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def compute_metrics(conf: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for tier in TIERS:
        tp = conf[tier][tier]
        fp = sum(conf[e][tier] for e in TIERS if e != tier)
        fn = sum(conf[tier][p] for p in TIERS if p != tier)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        support = sum(conf[tier].values())
        metrics[tier] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return metrics


def print_confusion(conf: Dict[str, Dict[str, int]]) -> None:
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("           fast  balanced  quality")
    for row in TIERS:
        print(
            f"{row:9} "
            f"{conf[row]['fast']:5} "
            f"{conf[row]['balanced']:9} "
            f"{conf[row]['quality']:8}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate adaptive router on labeled prompts.")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to labeled router eval dataset.")
    p.add_argument("--model-id", default=MODEL_ID, help="Tokenizer model id for token-length routing.")
    p.add_argument("--no-tokenizer", action="store_true", help="Route using word-count fallback.")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Path to write JSON report.")
    p.add_argument("--print-errors", type=int, default=10, help="Print top-N misroutes.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_path = Path(args.out)

    rows = load_dataset(dataset_path)
    if not rows:
        raise SystemExit(f"No valid rows found in dataset: {dataset_path}")

    tokenizer = None if args.no_tokenizer else AutoTokenizer.from_pretrained(args.model_id)
    conf = empty_confusion()
    errors = []

    for i, row in enumerate(rows, start=1):
        prompt = row["prompt"]
        expected = row["expected_tier"]
        decision = route(prompt, tokenizer=tokenizer)
        predicted = decision.tier
        conf[expected][predicted] += 1
        if expected != predicted:
            errors.append(
                {
                    "index": i,
                    "expected_tier": expected,
                    "predicted_tier": predicted,
                    "task_type": decision.task_type,
                    "confidence": round(decision.confidence, 3),
                    "prompt_len": decision.prompt_len,
                    "reason": decision.reason,
                    "prompt": prompt,
                }
            )

    total = len(rows)
    correct = sum(conf[t][t] for t in TIERS)
    accuracy = safe_div(correct, total)
    metrics = compute_metrics(conf)
    macro_f1 = safe_div(sum(metrics[t]["f1"] for t in TIERS), len(TIERS))

    print(f"Dataset: {dataset_path}")
    print(f"Total prompts: {total}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.4f}")
    print_confusion(conf)

    print("\nPer-tier metrics:")
    for t in TIERS:
        m = metrics[t]
        print(
            f"- {t:8} precision={m['precision']:.4f} "
            f"recall={m['recall']:.4f} f1={m['f1']:.4f} support={m['support']}"
        )

    if errors:
        print(f"\nTop {min(args.print_errors, len(errors))} misroutes:")
        for e in errors[: args.print_errors]:
            prompt_short = e["prompt"][:120] + ("..." if len(e["prompt"]) > 120 else "")
            print(
                f"- #{e['index']} expected={e['expected_tier']} predicted={e['predicted_tier']} "
                f"task={e['task_type']} conf={e['confidence']} len={e['prompt_len']}"
            )
            print(f"  prompt: {prompt_short}")
            print(f"  reason: {e['reason']}")
    else:
        print("\nNo misroutes in this dataset.")

    report = {
        "dataset": str(dataset_path),
        "model_id": None if args.no_tokenizer else args.model_id,
        "used_tokenizer": not args.no_tokenizer,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "confusion_matrix": conf,
        "per_tier_metrics": metrics,
        "errors": errors,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
