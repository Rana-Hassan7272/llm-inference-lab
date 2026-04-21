import argparse
import json
import math
import platform
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TIER_FILES = {
    "4-bit (fast)": RESULTS / "4bit_results.json",
    "8-bit (balanced)": RESULTS / "8bit_results.json",
    "FP16 (quality)": RESULTS / "fp16_results.json",
}

TIER_KEY = {
    "4-bit (fast)": "4bit",
    "8-bit (balanced)": "8bit",
    "FP16 (quality)": "fp16",
}

FACTUAL_REFS: Dict[str, List[str]] = {
    "what is the capital of france?": ["paris"],
    "what is the boiling point of water at sea level in celsius?": ["100 celsius", "100", "100 degrees celsius", "100 °c"],
    "who wrote the play hamlet?": ["william shakespeare", "shakespeare"],
    "what is 17 multiplied by 13?": ["221"],
    "what is the largest planet in our solar system?": ["jupiter"],
    "what is http status code 404?": ["not found", "404 not found"],
    "what does gpu stand for and why is it useful for ai?": ["graphics processing unit"],
    "what year did world war ii end?": ["1945"],
}

START_MARK = "<!-- ABLATION_TABLE_START -->"
END_MARK = "<!-- ABLATION_TABLE_END -->"
CONTROL_TOKENS_RE = re.compile(
    r"(\[/?(?:INST|USER|SYSTEM|ASSISTANT|SPEAKER|SQ|SURVEY|HEADER|BOTTOM|BUTTONS|CALL-TO-ACTION|CONTACT-INFO|CONTACT-ME|SCH)\])",
    flags=re.IGNORECASE,
)
MULTISPACE_RE = re.compile(r"\s+")


def load_results(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = CONTROL_TOKENS_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s\.\,\-\+\/]", " ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def clean_answer(answer: str, max_chars: int = 240) -> str:
    text = normalize_text(answer)
    if not text:
        return ""
    # Keep only first segment to reduce long degenerate continuation loops.
    parts = re.split(r"(?:\n|\.\s+|\?\s+|\!\s+)", text)
    first = parts[0].strip() if parts else text
    trimmed = first if len(first) >= 8 else text
    return trimmed[:max_chars].strip()


def tokenize(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", normalize_text(s), flags=re.UNICODE)


def bleu1_score(hypo: str, refs: List[str]) -> float:
    hyp_toks = tokenize(hypo)
    if not hyp_toks:
        return 0.0
    best = 0.0
    for ref in refs:
        ref_toks = tokenize(ref)
        ref_counts: Dict[str, int] = {}
        for t in ref_toks:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        match = 0
        used: Dict[str, int] = {}
        for t in hyp_toks:
            c = used.get(t, 0)
            if c < ref_counts.get(t, 0):
                match += 1
                used[t] = c + 1
        precision = match / len(hyp_toks)
        r = len(ref_toks)
        c = len(hyp_toks)
        bp = 1.0 if c > r else (0.0 if r == 0 else min(1.0, c / r))
        best = max(best, bp * precision)
    return best


def exact_match(pred: str, refs: List[str]) -> float:
    p = normalize_text(pred)
    for r in refs:
        if p == normalize_text(r):
            return 1.0
    return 0.0


def token_f1(pred: str, refs: List[str]) -> float:
    pred_toks = tokenize(pred)
    if not pred_toks:
        return 0.0
    best = 0.0
    for ref in refs:
        ref_toks = tokenize(ref)
        if not ref_toks:
            continue
        pred_counts: Dict[str, int] = {}
        ref_counts: Dict[str, int] = {}
        for t in pred_toks:
            pred_counts[t] = pred_counts.get(t, 0) + 1
        for t in ref_toks:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        overlap = 0
        for t, c in pred_counts.items():
            overlap += min(c, ref_counts.get(t, 0))
        if overlap == 0:
            continue
        precision = overlap / len(pred_toks)
        recall = overlap / len(ref_toks)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def build_ppl_model_for_tier(tier_key: str, model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"low_cpu_mem_usage": True}
    if tier_key == "fp16":
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if not torch.cuda.is_available() or BitsAndBytesConfig is None:
            raise RuntimeError(f"{tier_key} perplexity requires CUDA + BitsAndBytesConfig.")
        if tier_key == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif tier_key == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


def tier_perplexity(entries: List[Dict], tier_key: str, model_id: str) -> Optional[float]:
    try:
        tokenizer, model = build_ppl_model_for_tier(tier_key, model_id)
    except Exception:
        return None

    losses: List[float] = []
    with torch.no_grad():
        for e in entries:
            q = (e.get("question") or "").strip()
            a = clean_answer(e.get("answer") or "")
            if not q or not a:
                continue
            text = f"Question: {q}\nAnswer: {a}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss_val = float(outputs.loss.item())
            if math.isfinite(loss_val):
                losses.append(loss_val)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not losses:
        return None
    return float(math.exp(mean(losses)))


def should_compute_ppl_for_tier(do_ppl: bool, tier_key: str, ppl_tiers: set[str]) -> bool:
    if not do_ppl:
        return False
    return tier_key in ppl_tiers


def aggregate_tier(
    entries: List[Dict],
    do_ppl: bool,
    tier_key: str,
    model_id: str,
    ppl_tiers: set[str],
) -> Dict[str, float]:
    ttft = [float(e.get("ttft_ms", 0.0)) for e in entries if isinstance(e.get("ttft_ms", None), (int, float))]
    tps = [float(e.get("tok_per_sec", 0.0)) for e in entries if isinstance(e.get("tok_per_sec", None), (int, float))]
    vram = [float(e.get("mem_gb", 0.0)) for e in entries if isinstance(e.get("mem_gb", None), (int, float))]

    factual_em_scores: List[float] = []
    factual_f1_scores: List[float] = []
    bleu_scores: List[float] = []
    factual_count = 0

    for e in entries:
        q = normalize_text(e.get("question", ""))
        a = clean_answer(e.get("answer", ""))
        refs = FACTUAL_REFS.get(q)
        if not refs:
            continue
        factual_count += 1
        factual_em_scores.append(exact_match(a, refs))
        factual_f1_scores.append(token_f1(a, refs))
        bleu_scores.append(bleu1_score(a, refs))

    ppl = tier_perplexity(entries, tier_key, model_id) if should_compute_ppl_for_tier(do_ppl, tier_key, ppl_tiers) else None

    return {
        "vram_gb": round(mean(vram), 2),
        "ttft_ms": round(mean(ttft), 0),
        "tps": round(mean(tps), 2),
        "factual_em_pct": round(mean(factual_em_scores) * 100.0, 2) if factual_em_scores else 0.0,
        "factual_f1_pct": round(mean(factual_f1_scores) * 100.0, 2) if factual_f1_scores else 0.0,
        "bleu1_pct": round(mean(bleu_scores) * 100.0, 2) if bleu_scores else 0.0,
        "factual_coverage_pct": round((factual_count / len(entries)) * 100.0, 1) if entries else 0.0,
        "perplexity": round(ppl, 3) if ppl is not None else None,
    }


def build_table_data(do_ppl: bool, model_id: str, ppl_tiers: set[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tier, path in TIER_FILES.items():
        data = load_results(path)
        if not data:
            out[tier] = {}
            continue
        out[tier] = aggregate_tier(
            data,
            do_ppl=do_ppl,
            tier_key=TIER_KEY[tier],
            model_id=model_id,
            ppl_tiers=ppl_tiers,
        )
    return out


def write_manifest(table: Dict[str, Dict[str, float]]) -> Path:
    out_path = RESULTS / "ablation_table.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    return out_path


def render_markdown(table: Dict[str, Dict[str, float]], do_ppl: bool, ppl_tiers: set[str]) -> str:
    lines = []
    lines.append("### Ablation — Precision vs VRAM, TTFT, TPS, Factual Quality, and Perplexity")
    lines.append("")
    lines.append("Computed from `results/*_results.json`. Factual EM/F1 are measured on the factual subset; BLEU-1 is retained as a secondary signal only.")
    lines.append("")
    lines.append("| Tier | VRAM (GB) | Avg TTFT (ms) | Avg TPS (tok/s) | Factual EM (%) | Factual F1 (%) | BLEU-1 (0-100) | Perplexity (lower better) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    order = ["4-bit (fast)", "8-bit (balanced)", "FP16 (quality)"]
    for tier in order:
        row = table.get(tier, {})
        if not row:
            lines.append(f"| {tier} | — | — | — | — | — | — | — |")
            continue
        tier_key = TIER_KEY.get(tier, "")
        if isinstance(row.get("perplexity"), (int, float)):
            ppl = f"{row['perplexity']:.3f}"
        elif do_ppl and tier_key not in ppl_tiers:
            ppl = "skipped"
        else:
            ppl = "not computed" if do_ppl else "disabled"
        lines.append(
            f"| {tier} | {row['vram_gb']:.2f} | {row['ttft_ms']:.0f} | {row['tps']:.2f} | "
            f"{row['factual_em_pct']:.2f} | {row['factual_f1_pct']:.2f} | {row['bleu1_pct']:.2f} | {ppl} |"
        )
    lines.append("")
    covs = [table.get(t, {}).get("factual_coverage_pct") for t in order if table.get(t)]
    cov_avg = round(mean([c for c in covs if isinstance(c, (int, float))]), 1) if covs else 0.0
    lines.append(
        f"Notes: Factual coverage is ~{cov_avg}% (shared question subset). BLEU-1 is fragile on short references; "
        "prefer Factual EM/F1 and Perplexity for tier comparison."
    )
    return "\n".join(lines)


def update_readme(rendered: str) -> None:
    text = README.read_text(encoding="utf-8")
    if START_MARK in text and END_MARK in text:
        pre = text.split(START_MARK, 1)[0]
        post = text.split(END_MARK, 1)[1]
        new_text = f"{pre}{START_MARK}\n\n{rendered}\n\n{END_MARK}{post}"
    else:
        new_text = f"{text.rstrip()}\n\n{START_MARK}\n\n{rendered}\n\n{END_MARK}\n"
    README.write_text(new_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ablation quality table from benchmark outputs.")
    p.add_argument("--model-id", default=MODEL_ID, help="HF model id for perplexity computation.")
    p.add_argument(
        "--compute-perplexity",
        action="store_true",
        help="Compute tier-wise perplexity (can be slow; requires CUDA for 4-bit/8-bit).",
    )
    p.add_argument(
        "--ppl-tiers",
        default="auto",
        help="Comma-separated tiers for perplexity: auto|fp16|4bit,8bit,fp16",
    )
    return p.parse_args()


def resolve_ppl_tiers(ppl_tiers_arg: str) -> set[str]:
    value = (ppl_tiers_arg or "auto").strip().lower()
    valid = {"4bit", "8bit", "fp16"}
    if value == "auto":
        # Windows + quantized ppl is unstable on many setups; default to fp16 only.
        if platform.system().lower() == "windows":
            return {"fp16"}
        # On non-Windows keep all, but quantized tiers will still gracefully skip if unsupported.
        return {"4bit", "8bit", "fp16"}
    tiers = {t.strip() for t in value.split(",") if t.strip()}
    return {t for t in tiers if t in valid}


def main() -> None:
    args = parse_args()
    ppl_tiers = resolve_ppl_tiers(args.ppl_tiers)
    if args.compute_perplexity and not ppl_tiers:
        print("Perplexity requested but no valid tiers selected. Skipping perplexity.")
    elif args.compute_perplexity:
        print(f"Perplexity tiers: {sorted(ppl_tiers)}")

    table = build_table_data(
        do_ppl=args.compute_perplexity,
        model_id=args.model_id,
        ppl_tiers=ppl_tiers,
    )
    write_manifest(table)
    rendered = render_markdown(table, do_ppl=args.compute_perplexity, ppl_tiers=ppl_tiers)
    update_readme(rendered)
    print("Ablation table generated and README updated.")


if __name__ == "__main__":
    main()

