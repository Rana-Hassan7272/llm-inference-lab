"""
Phase 5 - Step 1: Benchmark Runner

One command orchestrates benchmark experiments and stores JSON artifacts.

Example:
    python benchmarks/runner.py --model tinyllama --modes fp16,8bit,4bit
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MODES = ["fp16", "8bit", "4bit"]
DEFAULT_MAX_NEW_TOKENS = 100

# Kept aligned with existing result file schema.
QUANT_QUESTIONS = [
    "What is the capital of France?",
    "Explain the difference between RAM and ROM.",
    "What causes rainbows to form?",
    "Who wrote the play Hamlet?",
    "What is photosynthesis? Explain briefly.",
    "What is 17 multiplied by 13?",
    "Name three programming languages and one use-case for each.",
    "What is the boiling point of water at sea level in Celsius?",
    "Explain what an API is in simple terms.",
    "What is the largest planet in our solar system?",
    "Define machine learning in two sentences.",
    "What is the purpose of unit testing?",
    "What is HTTP status code 404?",
    "Describe recursion in programming.",
    "What is the Pythagorean theorem?",
    "Explain the concept of overfitting in ML.",
    "What is the difference between supervised and unsupervised learning?",
    "What does GPU stand for and why is it useful for AI?",
    "Name three common Python data structures.",
    "What is Docker used for?",
]


CONTROL_TOKEN_RE = re.compile(
    r"(\[/?(?:INST|USER|SYSTEM|ASSISTANT|SPEAKER|SQ|SURVEY|HEADER|BOTTOM|BUTTONS|CALL-TO-ACTION|CONTACT-INFO|CONTACT-ME|SCH)\])",
    flags=re.IGNORECASE,
)


@dataclass
class StageResult:
    name: str
    status: str
    duration_sec: float
    output_files: list[str]
    message: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mem_gb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.memory_reserved() / (1024**3), 2)
    return 0.0


def _normalize_mode(mode: str) -> str:
    m = mode.strip().lower()
    aliases = {
        "fp16": "fp16",
        "16bit": "fp16",
        "8bit": "8bit",
        "8-bit": "8bit",
        "4bit": "4bit",
        "4-bit": "4bit",
    }
    if m not in aliases:
        raise ValueError(f"Unsupported mode '{mode}'. Allowed: fp16,8bit,4bit")
    return aliases[m]


def _resolve_model_arg(model_arg: str) -> str:
    model_aliases = {
        "tinyllama": DEFAULT_MODEL,
        "tinyllama-1.1b": DEFAULT_MODEL,
    }
    return model_aliases.get(model_arg.lower(), model_arg)


def _load_quantized_model(model_id: str, mode: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(ROOT / ".hf_cache"),
        local_files_only=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}

    if mode == "fp16":
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs["torch_dtype"] = dtype
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{mode} requires CUDA for reliable execution.")
        if BitsAndBytesConfig is None:
            raise RuntimeError(
                "bitsandbytes/quantization config unavailable. Install latest transformers + bitsandbytes."
            )
        if mode == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif mode == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(ROOT / ".hf_cache"),
        local_files_only=False,
        **kwargs,
    )
    # Avoid noisy warning: when both generation_config.max_length and max_new_tokens exist,
    # Transformers warns that max_new_tokens takes precedence.
    try:
        model.generation_config.max_length = None
    except Exception:
        pass
    if mode == "fp16":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    model.eval()
    return tokenizer, model


def _format_prompt(tokenizer, question: str, device: str) -> dict[str, torch.Tensor]:
    """
    Build inputs with chat template when available, else fallback to plain text.
    This reduces prompt-format mismatch artifacts in benchmark outputs.
    """
    text_prompt = f"Question: {question}\nAnswer:"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": question}]
            templated = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(templated, str) and templated.strip():
                text_prompt = templated
        except Exception:
            pass

    inputs = tokenizer(
        text_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def _clean_answer(answer: str) -> str:
    text = (answer or "").strip()
    text = CONTROL_TOKEN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove common echoed prefixes.
    for prefix in ("Answer:", "answer:", "Assistant:", "assistant:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    # Keep first sentence-ish chunk to reduce repeated loops from tiny models.
    parts = re.split(r"(?:\.\s+|\?\s+|\!\s+|\n+)", text)
    if parts and len(parts[0].strip()) >= 8:
        text = parts[0].strip()

    return text


def run_quantization_benchmark(
    model_id: str,
    mode: str,
    max_new_tokens: int,
    out_file: Path,
    question_limit: int | None = None,
) -> StageResult:
    start = time.perf_counter()
    mode_label = {"fp16": "FP16", "8bit": "8-bit", "4bit": "4-bit"}[mode]
    print(f"[quantization] Running {mode_label} benchmark ...")

    if mode in {"8bit", "4bit"} and not torch.cuda.is_available():
        return StageResult(
            name=f"quantization_{mode}",
            status="skipped",
            duration_sec=round(time.perf_counter() - start, 2),
            output_files=[],
            message=f"{mode} skipped: CUDA not available on this machine.",
        )

    try:
        tokenizer, model = _load_quantized_model(model_id, mode)
        device = next(model.parameters()).device
    except Exception as exc:
        return StageResult(
            name=f"quantization_{mode}",
            status="failed",
            duration_sec=round(time.perf_counter() - start, 2),
            output_files=[],
            message=str(exc),
        )

    records: list[dict[str, Any]] = []

    questions = QUANT_QUESTIONS[:question_limit] if question_limit else QUANT_QUESTIONS
    for idx, question in enumerate(questions, start=1):
        inputs = _format_prompt(tokenizer, question, str(device))
        in_len = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ttft = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t_ttft) * 1000

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

        new_tokens = max(output.shape[1] - in_len, 1)
        tok_per_sec = new_tokens / (total_ms / 1000) if total_ms > 0 else 0.0
        answer_raw = tokenizer.decode(output[0][in_len:], skip_special_tokens=True).strip()
        answer = _clean_answer(answer_raw)

        records.append(
            {
                "mode": mode_label,
                "question_id": idx,
                "question": question,
                "answer": answer,
                "ttft_ms": round(ttft_ms, 1),
                "tok_per_sec": round(tok_per_sec, 2),
                "mem_gb": _mem_gb(),
                "quality_score": None,
            }
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return StageResult(
        name=f"quantization_{mode}",
        status="ok",
        duration_sec=round(time.perf_counter() - start, 2),
        output_files=[str(out_file)],
        message=f"{len(records)} prompts benchmarked",
    )


def run_python_script(
    script_path: Path,
    args: list[str],
    cwd: Path,
    stage_name: str,
    expected_files: list[Path],
    skip_on_error: bool = False,
) -> StageResult:
    start = time.perf_counter()
    cwd.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script_path), *args]
    print(f"[runner] {stage_name}: {' '.join(cmd)}")

    child_env = dict(**__import__("os").environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        status = "ok"
        message = proc.stdout[-1200:].strip()
    except subprocess.CalledProcessError as exc:
        status = "skipped" if skip_on_error else "failed"
        message = (exc.stderr or exc.stdout or str(exc))[-1200:].strip()
        if not skip_on_error:
            return StageResult(
                name=stage_name,
                status=status,
                duration_sec=round(time.perf_counter() - start, 2),
                output_files=[],
                message=message,
            )

    outputs = [str(p) for p in expected_files if p.exists()]
    if status == "ok" and not outputs:
        status = "failed"
        message = "Stage finished but expected output files were not produced."

    return StageResult(
        name=stage_name,
        status=status,
        duration_sec=round(time.perf_counter() - start, 2),
        output_files=outputs,
        message=message,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 benchmark orchestration runner")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or alias (tinyllama)")
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated quantization modes: fp16,8bit,4bit",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--question-limit",
        type=int,
        default=0,
        help="Run only first N benchmark questions (0 = all 20).",
    )
    parser.add_argument("--skip-kv-cache", action="store_true")
    parser.add_argument("--skip-batching", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for compatible experiments")
    args = parser.parse_args()

    model_id = _resolve_model_arg(args.model)
    modes = [_normalize_mode(m) for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise ValueError("No valid modes provided.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at": _now_iso(),
        "model": model_id,
        "modes": modes,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "stages": [],
    }

    # Stage 1: quantization benchmark sweep
    quant_stage_ok = True
    for mode in modes:
        filename = {"fp16": "fp16_results.json", "8bit": "8bit_results.json", "4bit": "4bit_results.json"}[mode]
        stage = run_quantization_benchmark(
            model_id=model_id,
            mode=mode,
            max_new_tokens=args.max_new_tokens,
            out_file=RESULTS_DIR / filename,
            question_limit=(args.question_limit if args.question_limit > 0 else None),
        )
        run_manifest["stages"].append(asdict(stage))
        if stage.status != "ok":
            quant_stage_ok = False

    # Stage 2: kv-cache experiment
    if not args.skip_kv_cache:
        kv_dir = RESULTS_DIR / "kv_cache_experiment-results"
        kv_args = ["--model", model_id]
        if args.cpu:
            kv_args.append("--cpu")
        kv_stage = run_python_script(
            script_path=ROOT / "optimization" / "kv_cache_experiment.py",
            args=kv_args,
            cwd=kv_dir,
            stage_name="kv_cache",
            expected_files=[
                kv_dir / "kv_cache_results.json",
            ],
        )
        run_manifest["stages"].append(asdict(kv_stage))

    # Stage 3: static batching experiment
    if not args.skip_batching:
        batching_dir = RESULTS_DIR / "batching-results"
        batching_args = ["--model", model_id]
        if args.cpu:
            batching_args.append("--cpu")
        batching_stage = run_python_script(
            script_path=ROOT / "optimization" / "batching.py",
            args=batching_args,
            cwd=batching_dir,
            stage_name="batching",
            expected_files=[
                batching_dir / "batching_results.json",
            ],
        )
        run_manifest["stages"].append(asdict(batching_stage))

    # Stage 4: vLLM comparison (optional / platform constrained)
    if not args.skip_vllm:
        vllm_dir = RESULTS_DIR / "vllm"
        if platform.system().lower() != "linux":
            run_manifest["stages"].append(
                asdict(
                    StageResult(
                        name="vllm",
                        status="skipped",
                        duration_sec=0.0,
                        output_files=[],
                        message="vLLM stage skipped: requires Linux + NVIDIA GPU.",
                    )
                )
            )
        else:
            vllm_stage = run_python_script(
                script_path=ROOT / "benchmarks" / "batching_comparison-vllm.py",
                args=["--model", model_id, "--compare-file", str(RESULTS_DIR / "batching-results" / "batching_results.json")],
                cwd=vllm_dir,
                stage_name="vllm",
                expected_files=[vllm_dir / "vllm_results.json"],
                skip_on_error=True,
            )
            run_manifest["stages"].append(asdict(vllm_stage))

    run_manifest["finished_at"] = _now_iso()
    run_manifest["success"] = quant_stage_ok and all(
        s["status"] in ("ok", "skipped") for s in run_manifest["stages"]
    )

    manifest_path = RESULTS_DIR / f"benchmark_run_{run_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[runner] Manifest saved: {manifest_path}")
    for st in run_manifest["stages"]:
        print(f"  - {st['name']}: {st['status']} ({st['duration_sec']}s)")
    print(f"[runner] success={run_manifest['success']}")
    return 0 if run_manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

