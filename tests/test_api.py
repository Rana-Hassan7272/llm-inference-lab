import contextlib
import json
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

# Ensure project root is importable when pytest runs from different environments.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.app as api_app


class DummyTokenizer:
    eos_token_id = 2

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        token_count = max(4, len(text.split()))
        ids = torch.ones((1, token_count), dtype=torch.long)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "shape"):
            n = int(ids.shape[0]) if len(ids.shape) == 1 else int(ids.shape[-1])
        else:
            n = len(ids)
        return "ok " * max(1, n // 2)

    def encode(self, text):
        return [1] * max(1, len(text.split()))


class DummyStreamer:
    def __init__(self, *args, **kwargs):
        self._tokens = []

    def __iter__(self):
        return iter(self._tokens)


class DummyModel:
    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        max_new_tokens = int(kwargs.get("max_new_tokens", 1))
        streamer = kwargs.get("streamer")

        if streamer is not None:
            # Simulate streamed chunks.
            streamer._tokens = ["hello", " world"]
            return None

        bs, in_len = input_ids.shape
        extra = torch.ones((bs, max_new_tokens), dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)


class DummyManager:
    def __init__(self):
        self.model_id = "dummy-model"
        self.device = "cpu"
        self._tiers = {}
        self._tokenizer = DummyTokenizer()

    @property
    def tokenizer(self):
        return self._tokenizer

    def get(self, tier):
        self._tiers[tier] = True
        return DummyModel(), self._tokenizer

    def get_lock(self, tier):
        return contextlib.nullcontext()

    def status(self):
        return {"model_id": self.model_id, "device": self.device, "loaded_tiers": list(self._tiers.keys())}

    def total_requests(self):
        return 0

    def unload_all(self):
        return None


def _dummy_route(prompt, tokenizer=None):
    return api_app.RoutingDecision(
        tier="fast",
        precision="4-bit",
        reason="test route",
        prompt_len=10,
        task_type="simple",
        confidence=0.9,
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_app, "manager", DummyManager())
    monkeypatch.setattr(api_app, "route", _dummy_route)
    monkeypatch.setattr(api_app, "TextIteratorStreamer", DummyStreamer)
    with TestClient(api_app.app) as c:
        yield c


def test_health_format(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "model_id" in body
    assert "device" in body
    assert "loaded_tiers" in body


def test_generate_format(client):
    payload = {"prompt": "What is the capital of France?", "max_tokens": 20, "temperature": 0.0}
    r = client.post("/generate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert "routing" in body
    assert "tok_per_sec" in body
    assert "ttft_ms" in body
    assert "total_ms" in body
    assert "mem_gb" in body


def test_benchmark_format(client):
    r = client.get("/benchmark/fast?max_tokens=20")
    assert r.status_code == 200
    body = r.json()
    assert "model_id" in body
    assert "benchmark" in body
    assert isinstance(body["benchmark"], list)
    assert len(body["benchmark"]) >= 1


def test_streaming_basic_shape(client):
    payload = {"prompt": "Explain recursion briefly", "max_tokens": 16, "temperature": 0.0}
    with client.stream("POST", "/generate/stream", json=payload) as r:
        assert r.status_code == 200
        chunks = "".join([part for part in r.iter_text()])

    # Ensure SSE format includes token events and final done marker.
    assert "data:" in chunks
    assert '"token"' in chunks
    assert '"done": true' in chunks

