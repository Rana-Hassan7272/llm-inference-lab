"""
api/app.py
===========
Phase 4 — Steps 1 + 2 + 3 + 4: Complete FastAPI Inference Server

Endpoints:
  POST /generate              — standard (blocking) generation with adaptive routing
  POST /generate/stream       — streaming generation (token by token, like ChatGPT)
  GET  /benchmark/{model}     — run quick benchmark for a model config
  GET  /health                — server + model status
  GET  /status                — detailed model manager status
  GET  /router/explain        — show routing decision for a prompt (no generation)
  GET  /routing-log           — last N routing decisions from the log file

Architecture:
  Request → adaptive_router.route() → ModelManager.get(tier) → generation
  Every request: routing decision appended to results/routing_log.jsonl

HOW TO RUN:
  # Install deps first:
  pip install fastapi uvicorn transformers torch accelerate bitsandbytes

  # Start server (development):
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

  # Start server (production):
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1
  # Note: workers=1 required — GPU models are not fork-safe.

  # In Colab:
  import subprocess, threading
  def run(): subprocess.run(["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"])
  threading.Thread(target=run, daemon=True).start()

QUICK TEST (once server is running):
  curl -X POST http://localhost:8000/generate \
       -H "Content-Type: application/json" \
       -d '{"prompt": "What is the capital of France?", "max_tokens": 100}'

  curl -X POST http://localhost:8000/generate/stream \
       -H "Content-Type: application/json" \
       -d '{"prompt": "Explain recursion in programming.", "max_tokens": 200}'

  curl http://localhost:8000/health
  curl "http://localhost:8000/router/explain?prompt=Write+a+Python+function"
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import TextIteratorStreamer

# ── Local imports — adjust path if running from project root ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.model_manager import ModelManager
from inference.adaptive_router import route, RoutingDecision

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MODEL_ID         = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MAX_LOADED_TIERS = int(os.getenv("MAX_LOADED_TIERS", "3"))   # 1 for large models
LOG_DIR          = Path(os.getenv("LOG_DIR", "results"))
ROUTING_LOG_PATH = LOG_DIR / "routing_log.jsonl"             # JSONL: one entry per line

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  APP + MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="LLM Inference API",
    description=(
        "Production-ready LLM inference server with adaptive routing. "
        "Routes requests to 4-bit / 8-bit / FP16 model tiers based on "
        "prompt complexity. Supports streaming responses."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL MODEL MANAGER (one instance, shared across all requests)
# ══════════════════════════════════════════════════════════════════════════════

manager = ModelManager(model_id=MODEL_ID, max_loaded_tiers=MAX_LOADED_TIERS)

# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class GenerateRequest(BaseModel):
    prompt:     str            = Field(...,   min_length=1, max_length=4096,
                                       description="The input prompt to generate from")
    max_tokens: int            = Field(200,   ge=1, le=2048,
                                       description="Maximum new tokens to generate")
    temperature: float         = Field(0.0,  ge=0.0, le=2.0,
                                       description="Sampling temperature (0 = greedy)")
    force_tier: Optional[str]  = Field(None,
                                       description="Override routing: 'fast' | 'balanced' | 'quality'")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain how transformers work in machine learning.",
                "max_tokens": 200,
                "temperature": 0.0,
            }
        }


class RoutingInfo(BaseModel):
    tier:       str
    precision:  str
    reason:     str
    prompt_len: int
    task_type:  str
    confidence: float


class GenerateResponse(BaseModel):
    text:         str
    routing:      RoutingInfo
    tok_per_sec:  float
    ttft_ms:      float
    total_ms:     float
    mem_gb:       float
    model_id:     str
    timestamp:    str


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTING LOG  (JSONL — append-only, crash-safe)
# ══════════════════════════════════════════════════════════════════════════════

_log_lock = threading.Lock()


def append_routing_log(entry: dict) -> None:
    """Append one routing decision to the JSONL log file. Thread-safe."""
    with _log_lock:
        with open(ROUTING_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_routing_log(last_n: int = 50) -> list[dict]:
    """Read the last N entries from the routing log."""
    if not ROUTING_LOG_PATH.exists():
        return []
    lines = ROUTING_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-last_n:]]


# ══════════════════════════════════════════════════════════════════════════════
#  CORE GENERATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_inputs(prompt: str, tokenizer, device: str) -> tuple:
    """Tokenize prompt and return (inputs_dict, input_token_count)."""
    inputs   = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs   = {k: v.to(device) for k, v in inputs.items()}
    in_len   = inputs["input_ids"].shape[1]
    return inputs, in_len


def _make_generate_kwargs(
    inputs: dict,
    max_tokens: int,
    temperature: float,
    tokenizer,
    streamer=None,
) -> dict:
    """Build model.generate() kwargs. Shared by stream and non-stream paths."""
    kwargs = {
        **inputs,
        "max_new_tokens":  max_tokens,
        "do_sample":       temperature > 0.0,
        "temperature":     temperature if temperature > 0.0 else 1.0,
        "pad_token_id":    tokenizer.eos_token_id,
        "use_cache":       True,
    }
    if streamer is not None:
        kwargs["streamer"] = streamer
    return kwargs


def _build_log_entry(
    prompt: str,
    decision: RoutingDecision,
    tok_per_sec: float,
    ttft_ms: float,
    total_ms: float,
    mem_gb: float,
    streaming: bool,
) -> dict:
    return {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "prompt":      prompt[:200],          # truncate long prompts in log
        "streaming":   streaming,
        "routing": {
            "tier":       decision.tier,
            "precision":  decision.precision,
            "reason":     decision.reason,
            "prompt_len": decision.prompt_len,
            "task_type":  decision.task_type,
            "confidence": round(decision.confidence, 3),
        },
        "metrics": {
            "tok_per_sec": round(tok_per_sec, 2),
            "ttft_ms":     round(ttft_ms, 1),
            "total_ms":    round(total_ms, 1),
            "mem_gb":      round(mem_gb, 3),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1 — POST /generate  (blocking)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate(req: GenerateRequest):
    """
    Generate text from a prompt. Uses adaptive routing to select the
    optimal model precision tier (4-bit / 8-bit / FP16) automatically.

    Set force_tier to override routing: "fast" | "balanced" | "quality"
    """
    # ── Route ─────────────────────────────────────────────────────────────────
    tok      = manager.tokenizer
    decision = route(req.prompt, tokenizer=tok)

    if req.force_tier:
        if req.force_tier not in ("fast", "balanced", "quality"):
            raise HTTPException(status_code=400,
                                detail="force_tier must be 'fast', 'balanced', or 'quality'")
        decision.tier      = req.force_tier
        decision.precision = {"fast": "4-bit", "balanced": "8-bit", "quality": "FP16"}[req.force_tier]
        decision.reason    = f"Forced by caller to {req.force_tier} tier"

    logger.info("POST /generate | tier=%s | prompt=%.60s...", decision.tier, req.prompt)

    # ── Load model + tokenize ─────────────────────────────────────────────────
    try:
        model, tokenizer = manager.get(decision.tier)
    except Exception as e:
        logger.error("Model load failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Model load failed: {e}")

    device  = next(model.parameters()).device
    inputs, in_len = _prepare_inputs(req.prompt, tokenizer, str(device))

    # ── Time to first token ───────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_ttft = time.perf_counter()

    with manager.get_lock(decision.tier):
        with torch.no_grad():
            _ = model.generate(
                **inputs, max_new_tokens=1,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t_ttft) * 1000

        # ── Full generation ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        kwargs = _make_generate_kwargs(inputs, req.max_tokens, req.temperature, tokenizer)
        with torch.no_grad():
            output = model.generate(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

    new_tokens  = output.shape[1] - in_len
    tok_per_sec = new_tokens / (total_ms / 1000) if total_ms > 0 else 0
    text        = tokenizer.decode(output[0][in_len:], skip_special_tokens=True).strip()
    mem_gb      = ModelManager._vram_used_gb()

    # ── Log ───────────────────────────────────────────────────────────────────
    append_routing_log(_build_log_entry(
        req.prompt, decision, tok_per_sec, ttft_ms, total_ms, mem_gb,
        streaming=False
    ))

    return GenerateResponse(
        text=text,
        routing=RoutingInfo(**{
            "tier":       decision.tier,
            "precision":  decision.precision,
            "reason":     decision.reason,
            "prompt_len": decision.prompt_len,
            "task_type":  decision.task_type,
            "confidence": decision.confidence,
        }),
        tok_per_sec=round(tok_per_sec, 2),
        ttft_ms=round(ttft_ms, 1),
        total_ms=round(total_ms, 1),
        mem_gb=round(mem_gb, 3),
        model_id=manager.model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 2 — POST /generate/stream  (token-by-token streaming)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/generate/stream", tags=["inference"])
async def generate_stream(req: GenerateRequest):
    """
    Stream generated tokens one by one — exactly like ChatGPT.

    Response format: Server-Sent Events (text/event-stream)
      data: {"token": "Hello"}
      data: {"token": " world"}
      data: {"token": ""}       ← empty token = stream finished
      data: {"done": true, "tok_per_sec": 42.1, "routing": {...}}

    How to consume in JavaScript:
      const res  = await fetch('/generate/stream', {method:'POST', body:...});
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        const lines = decoder.decode(value).split('\\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const obj = JSON.parse(line.slice(6));
            if (obj.token !== undefined) process(obj.token);
          }
        }
      }

    How it works internally:
      TextIteratorStreamer runs in a background thread and feeds tokens
      into a queue. The FastAPI async generator reads from the queue
      and yields SSE lines. This avoids blocking the async event loop.
    """
    # ── Route ─────────────────────────────────────────────────────────────────
    tok      = manager.tokenizer
    decision = route(req.prompt, tokenizer=tok)

    if req.force_tier:
        if req.force_tier not in ("fast", "balanced", "quality"):
            raise HTTPException(status_code=400,
                                detail="force_tier must be 'fast', 'balanced', or 'quality'")
        decision.tier      = req.force_tier
        decision.precision = {"fast": "4-bit", "balanced": "8-bit", "quality": "FP16"}[req.force_tier]
        decision.reason    = f"Forced by caller to {req.force_tier} tier"

    logger.info("POST /generate/stream | tier=%s | prompt=%.60s...", decision.tier, req.prompt)

    try:
        model, tokenizer = manager.get(decision.tier)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model load failed: {e}")

    device = next(model.parameters()).device
    inputs, in_len = _prepare_inputs(req.prompt, tokenizer, str(device))

    # ── Build streamer ─────────────────────────────────────────────────────────
    # TextIteratorStreamer: tokens are decoded and placed in an internal queue.
    # skip_prompt=True: we only stream the generated part, not the input back.
    # skip_special_tokens=True: no [EOS] or [PAD] in the stream.
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60.0,
    )

    gen_kwargs = _make_generate_kwargs(
        inputs, req.max_tokens, req.temperature, tokenizer, streamer=streamer
    )

    # ── State shared between generation thread and streaming generator ─────────
    gen_state = {
        "start_time":  None,
        "ttft_time":   None,
        "token_count": 0,
        "error":       None,
    }

    # ── Generation runs in a background thread ─────────────────────────────────
    # Why a thread? model.generate() is synchronous and blocks.
    # Running it in a thread lets FastAPI's async event loop keep serving
    # other requests while tokens are being generated.
    def _generate_thread():
        gen_state["start_time"] = time.perf_counter()
        try:
            with manager.get_lock(decision.tier):
                with torch.no_grad():
                    model.generate(**gen_kwargs)
        except Exception as e:
            gen_state["error"] = str(e)
            logger.error("Generation thread error: %s", e)

    thread = threading.Thread(target=_generate_thread, daemon=True)
    thread.start()

    # ── Async generator: reads tokens from streamer, yields SSE lines ──────────
    async def token_generator() -> AsyncIterator[str]:
        token_count   = 0
        start_time    = time.perf_counter()
        ttft_recorded = False

        # SSE preamble: send routing decision immediately so the client
        # knows which model is being used before any tokens arrive.
        preamble = {
            "routing": {
                "tier":      decision.tier,
                "precision": decision.precision,
                "reason":    decision.reason,
                "task_type": decision.task_type,
            }
        }
        yield f"data: {json.dumps(preamble)}\n\n"

        for token_text in streamer:
            if gen_state["error"]:
                yield f"data: {json.dumps({'error': gen_state['error']})}\n\n"
                break

            if not ttft_recorded:
                ttft_ms = (time.perf_counter() - start_time) * 1000
                ttft_recorded = True
            else:
                ttft_ms = 0.0

            token_count += 1
            yield f"data: {json.dumps({'token': token_text})}\n\n"

        # ── Final metadata event ───────────────────────────────────────────────
        total_ms    = (time.perf_counter() - start_time) * 1000
        tok_per_sec = token_count / (total_ms / 1000) if total_ms > 0 else 0
        mem_gb      = ModelManager._vram_used_gb()

        done_payload = {
            "done":        True,
            "tok_per_sec": round(tok_per_sec, 2),
            "total_ms":    round(total_ms, 1),
            "token_count": token_count,
            "mem_gb":      round(mem_gb, 3),
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

        # ── Log ───────────────────────────────────────────────────────────────
        # Approximate ttft for log (streamer doesn't expose per-token timing)
        approx_ttft = (total_ms / token_count) if token_count > 0 else total_ms
        append_routing_log(_build_log_entry(
            req.prompt, decision, tok_per_sec, approx_ttft, total_ms, mem_gb,
            streaming=True
        ))

        logger.info(
            "Stream complete | tier=%s | tokens=%d | tok/s=%.1f | total=%.0fms",
            decision.tier, token_count, tok_per_sec, total_ms
        )

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",     # disables nginx buffering (important!)
            "Connection":     "keep-alive",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 3 — GET /benchmark/{model}
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_PROMPTS_QUICK = [
    ("What is the capital of France?",                 "fast"),
    ("Explain how the KV cache works in LLM inference.", "balanced"),
    (
        "Analyze the trade-offs between microservices and monolithic "
        "architecture for a high-traffic web application.",
        "quality"
    ),
]


@app.get("/benchmark/{model_tier}", tags=["benchmarks"])
async def benchmark(model_tier: str, max_tokens: int = 100):
    """
    Run a quick 3-prompt benchmark for a specific tier.
    model_tier: "fast" | "balanced" | "quality" | "all"
    """
    tiers = ["fast", "balanced", "quality"] if model_tier == "all" else [model_tier]
    for t in tiers:
        if t not in ("fast", "balanced", "quality"):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tier {t!r}. Use: fast, balanced, quality, all"
            )

    results = []
    for tier in tiers:
        tier_results = []
        try:
            model, tokenizer = manager.get(tier)
        except Exception as e:
            results.append({"tier": tier, "error": str(e)})
            continue

        device = next(model.parameters()).device

        for prompt, _ in BENCHMARK_PROMPTS_QUICK:
            inputs, in_len = _prepare_inputs(prompt, tokenizer, str(device))

            t0 = time.perf_counter()
            with manager.get_lock(tier):
                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=max_tokens,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id
                    )
            total_ms    = (time.perf_counter() - t0) * 1000
            new_tokens  = output.shape[1] - in_len
            tok_per_sec = new_tokens / (total_ms / 1000) if total_ms > 0 else 0

            tier_results.append({
                "prompt":      prompt[:60] + "...",
                "tok_per_sec": round(tok_per_sec, 2),
                "total_ms":    round(total_ms, 1),
                "new_tokens":  int(new_tokens),
            })

        avg_tps = sum(r["tok_per_sec"] for r in tier_results) / len(tier_results)
        results.append({
            "tier":       tier,
            "precision":  {"fast": "4-bit", "balanced": "8-bit", "quality": "FP16"}[tier],
            "avg_tok_per_sec": round(avg_tps, 2),
            "mem_gb":     round(ModelManager._vram_used_gb(), 3),
            "prompts":    tier_results,
        })

    return {"model_id": manager.model_id, "benchmark": results}


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 4 — GET /health
# ══════════════════════════════════════════════════════════════════════════════

_server_start = time.time()


@app.get("/health", tags=["system"])
async def health():
    """
    Quick health check. Returns server uptime, GPU status, loaded model tiers.
    """
    uptime_s = time.time() - _server_start
    return {
        "status":    "ok",
        "uptime_s":  round(uptime_s, 1),
        "model_id":  manager.model_id,
        "device":    manager.device,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name":  torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_tiers": list(manager._tiers.keys()),
        "total_requests": manager.total_requests(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 5 — GET /status  (detailed)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/status", tags=["system"])
async def status():
    """Detailed model manager status including VRAM usage per tier."""
    return manager.status()


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 6 — GET /router/explain  (routing decision, no generation)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/router/explain", tags=["routing"])
async def router_explain(prompt: str):
    """
    Show which tier the router would select for a prompt — without generating.
    Fast and useful for demos and debugging.

    Example:
      GET /router/explain?prompt=Write+a+Python+sorting+function
    """
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    decision = route(prompt, tokenizer=manager.tokenizer if manager._tokenizer else None)
    return {
        "prompt":     prompt,
        "tier":       decision.tier,
        "precision":  decision.precision,
        "reason":     decision.reason,
        "task_type":  decision.task_type,
        "confidence": round(decision.confidence, 3),
        "prompt_len": decision.prompt_len,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 7 — GET /routing-log
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/routing-log", tags=["routing"])
async def routing_log(last_n: int = 20):
    """Return the last N routing decisions from the log file."""
    entries = read_routing_log(last_n=last_n)
    return {
        "count":   len(entries),
        "log_path": str(ROUTING_LOG_PATH),
        "entries": entries,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP / SHUTDOWN EVENTS
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def on_startup():
    logger.info("=" * 60)
    logger.info("LLM Inference API starting up")
    logger.info("  Model   : %s", MODEL_ID)
    logger.info("  Device  : %s", manager.device)
    logger.info("  Log dir : %s", LOG_DIR)
    logger.info("  Routing log: %s", ROUTING_LOG_PATH)
    if torch.cuda.is_available():
        logger.info(
            "  GPU     : %s | %.1f GB VRAM",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )
    logger.info("  Preloading tokenizer...")
    _ = manager.tokenizer    # warm up tokenizer at startup, not on first request
    logger.info("  Tokenizer ready. Models load lazily on first request.")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down — freeing GPU memory...")
    manager.unload_all()
    logger.info("Shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT (python api/app.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,    # reload=True only in development (no GPU models)
        log_level="info",
    )