"""
Locust user behavior for Phase 5 Step 2 load testing.

Targets:
- POST /generate (primary throughput/latency benchmark)
- GET  /health (light health probe)
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task


PROMPTS = [
    "What is the capital of France?",
    "Explain the difference between RAM and ROM.",
    "Describe what machine learning is in simple terms.",
    "What is an API and why is it useful?",
    "Explain what overfitting means in ML.",
    "Name three Python data structures and one use-case each.",
    "What causes rainbows to form?",
    "Summarize the purpose of unit tests in software projects.",
]

FORCE_TIER = os.getenv("LOADTEST_FORCE_TIER", "").strip().lower()
if FORCE_TIER not in {"", "fast", "balanced", "quality"}:
    raise ValueError("LOADTEST_FORCE_TIER must be one of: fast, balanced, quality")

MAX_TOKENS = int(os.getenv("LOADTEST_MAX_TOKENS", "64"))
TEMPERATURE = float(os.getenv("LOADTEST_TEMPERATURE", "0.0"))


class InferenceUser(HttpUser):
    # Small think-time to model real request pacing and avoid pure flood mode.
    wait_time = between(0.2, 0.8)

    @task(5)
    def generate(self) -> None:
        payload = {
            "prompt": random.choice(PROMPTS),
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        if FORCE_TIER:
            payload["force_tier"] = FORCE_TIER

        with self.client.post("/generate", json=payload, catch_response=True, name="POST /generate") as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            try:
                body = resp.json()
            except Exception:
                resp.failure("Response is not valid JSON")
                return

            required_fields = ("text", "routing", "tok_per_sec", "ttft_ms", "total_ms", "mem_gb", "model_id")
            if not all(field in body for field in required_fields):
                resp.failure("Missing required fields in /generate response")
                return

            routing = body.get("routing", {})
            if "tier" not in routing or "precision" not in routing:
                resp.failure("Missing routing data in /generate response")
                return
            resp.success()

    @task(1)
    def health(self) -> None:
        with self.client.get("/health", catch_response=True, name="GET /health") as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            resp.success()

