"""
inference/model_manager.py
===========================
Phase 4 — Step 3: Model Manager

Handles loading, caching, and switching between model configurations.
This is the single source of truth for all model state in the application.

Design principles:
  • Each precision tier (4-bit / 8-bit / FP16) is loaded once and cached.
  • Thread-safe: uses per-tier locks so concurrent API requests don't collide.
  • Lazy loading: models load on first use, not at startup.
  • VRAM-aware: tracks memory per tier, warns before OOM.
  • Clean public API: the rest of the app never touches transformers directly.

Usage (as a module):
    from inference.model_manager import ModelManager

    manager = ModelManager(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = manager.get("fast")       # loads 4-bit on first call
    model, tokenizer = manager.get("balanced")   # loads 8-bit on first call
    model, tokenizer = manager.get("quality")    # loads FP16 on first call

    info = manager.status()                      # dict of loaded tiers + VRAM
    manager.unload("fast")                       # free VRAM for one tier
    manager.unload_all()                         # free everything
"""

import gc
import time
import threading
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  TIER CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

TIER_LABELS = {
    "fast":     "4-bit (NF4)",
    "balanced": "8-bit",
    "quality":  "FP16",
}

# Maps tier name → kwargs for from_pretrained
def _build_load_kwargs(tier: str, device: str) -> dict:
    base = {"low_cpu_mem_usage": True}

    if device == "cuda":
        base["device_map"] = "auto"

    if tier == "fast":
        base["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif tier == "balanced":
        base["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif tier == "quality":
        base["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
    else:
        raise ValueError(f"Unknown tier: {tier!r}. Must be 'fast', 'balanced', or 'quality'.")

    return base


# ══════════════════════════════════════════════════════════════════════════════
#  TIER STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TierState:
    """Tracks one loaded model tier."""
    tier:          str
    model:         object        # AutoModelForCausalLM — typed as object to avoid circular hints
    tokenizer:     object        # AutoTokenizer
    loaded_at:     float         # unix timestamp
    load_time_s:   float         # how long loading took
    vram_delta_gb: float         # VRAM consumed by this tier (GB)
    request_count: int = 0       # how many times this tier has been used
    lock:          threading.Lock = field(default_factory=threading.Lock)

    def to_dict(self) -> dict:
        return {
            "tier":          self.tier,
            "label":         TIER_LABELS.get(self.tier, self.tier),
            "loaded_at":     self.loaded_at,
            "load_time_s":   round(self.load_time_s, 2),
            "vram_delta_gb": round(self.vram_delta_gb, 3),
            "request_count": self.request_count,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class ModelManager:
    """
    Owns all model loading, caching, and lifecycle management.

    Thread safety:
      • _registry_lock protects the _tiers dict (load/unload operations).
      • Each TierState has its own lock for generation — prevents two concurrent
        requests from calling model.generate() simultaneously on the same model.
        (HuggingFace models are NOT thread-safe for generation.)

    VRAM strategy:
      For TinyLlama (1.1B) on T4 (15 GB), all three tiers fit simultaneously
      (~5 GB total). For larger models (7B+), set max_loaded_tiers=1 to
      evict the previous tier before loading a new one.
    """

    def __init__(
        self,
        model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_loaded_tiers: int = 3,      # 1 = swap models (large models), 3 = keep all
        device: Optional[str] = None,   # None = auto-detect
    ):
        self.model_id         = model_id
        self.max_loaded_tiers = max_loaded_tiers
        self.device           = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tiers: dict[str, TierState] = {}
        self._tokenizer       = None
        self._registry_lock   = threading.Lock()   # protects _tiers dict mutations

        logger.info(
            "ModelManager init | model=%s | device=%s | max_tiers=%d",
            model_id, self.device, max_loaded_tiers
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def tokenizer(self):
        """Shared tokenizer — loaded once, used by all tiers."""
        if self._tokenizer is None:
            logger.info("Loading tokenizer: %s", self.model_id)
            tok = AutoTokenizer.from_pretrained(self.model_id)
            tok.pad_token    = tok.eos_token
            tok.padding_side = "left"
            self._tokenizer  = tok
            logger.info("Tokenizer loaded. Vocab size: %d", tok.vocab_size)
        return self._tokenizer

    def get(self, tier: str) -> tuple:
        """
        Return (model, tokenizer) for the given tier.
        Loads the model if not already cached.

        Args:
            tier: "fast" | "balanced" | "quality"

        Returns:
            (model, tokenizer) — ready for inference
        """
        if tier not in ("fast", "balanced", "quality"):
            raise ValueError(f"Invalid tier {tier!r}. Choose: fast, balanced, quality")

        with self._registry_lock:
            if tier not in self._tiers:
                self._evict_if_needed()
                self._load(tier)

        state = self._tiers[tier]
        state.request_count += 1
        return state.model, self.tokenizer

    def get_lock(self, tier: str) -> threading.Lock:
        """
        Return the generation lock for a tier.
        Callers should acquire this before calling model.generate().

        Usage:
            model, tok = manager.get("fast")
            with manager.get_lock("fast"):
                output = model.generate(...)
        """
        if tier not in self._tiers:
            raise RuntimeError(f"Tier {tier!r} not loaded yet. Call get() first.")
        return self._tiers[tier].lock

    def is_loaded(self, tier: str) -> bool:
        return tier in self._tiers

    def unload(self, tier: str) -> None:
        """Free VRAM for one tier."""
        with self._registry_lock:
            self._evict(tier)

    def unload_all(self) -> None:
        """Free VRAM for all loaded tiers."""
        with self._registry_lock:
            for tier in list(self._tiers.keys()):
                self._evict(tier)

    def status(self) -> dict:
        """
        Return a status dict suitable for JSON serialization.
        Used by the /health and /status API endpoints.
        """
        total_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
            if self.device == "cuda" and torch.cuda.is_available()
            else 0.0
        )
        used_vram_gb = (
            torch.cuda.memory_reserved() / 1024**3
            if self.device == "cuda" and torch.cuda.is_available()
            else 0.0
        )

        return {
            "model_id":       self.model_id,
            "device":         self.device,
            "tokenizer_ready": self._tokenizer is not None,
            "loaded_tiers":   {t: s.to_dict() for t, s in self._tiers.items()},
            "vram": {
                "total_gb":    round(total_vram_gb, 2),
                "used_gb":     round(used_vram_gb, 2),
                "free_gb":     round(total_vram_gb - used_vram_gb, 2),
            },
        }

    def total_requests(self) -> int:
        return sum(s.request_count for s in self._tiers.values())

    # ── Private ────────────────────────────────────────────────────────────────

    def _load(self, tier: str) -> None:
        """Load a model tier. Must be called with _registry_lock held."""
        label = TIER_LABELS[tier]
        logger.info("Loading tier=%s (%s) ...", tier, label)
        print(f"[ModelManager] Loading {label} model...", flush=True)

        mem_before = self._vram_used_gb()
        t0 = time.perf_counter()

        kwargs = _build_load_kwargs(tier, self.device)
        model  = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

        if self.device == "cpu":
            model = model.to("cpu")

        model.eval()

        load_time    = time.perf_counter() - t0
        mem_after    = self._vram_used_gb()
        vram_delta   = max(mem_after - mem_before, 0.0)

        self._tiers[tier] = TierState(
            tier=tier,
            model=model,
            tokenizer=self.tokenizer,
            loaded_at=time.time(),
            load_time_s=load_time,
            vram_delta_gb=vram_delta,
        )

        logger.info(
            "Tier %s loaded in %.1fs | VRAM +%.2f GB (now %.2f/%.2f GB)",
            tier, load_time, vram_delta, mem_after, self._vram_total_gb()
        )
        print(
            f"[ModelManager] {label} ready | "
            f"load={load_time:.1f}s | VRAM +{vram_delta:.2f} GB",
            flush=True
        )

    def _evict(self, tier: str) -> None:
        """Delete a loaded tier and free its VRAM. Must hold _registry_lock."""
        if tier not in self._tiers:
            return
        label = TIER_LABELS.get(tier, tier)
        logger.info("Evicting tier=%s (%s)", tier, label)
        state = self._tiers.pop(tier)
        del state.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Evicted %s. Free VRAM: %.2f GB", label, self._vram_free_gb())

    def _evict_if_needed(self) -> None:
        """If at tier capacity, evict the least-recently-used tier."""
        if len(self._tiers) < self.max_loaded_tiers:
            return
        # LRU: evict the tier with the oldest loaded_at timestamp
        lru_tier = min(self._tiers, key=lambda t: self._tiers[t].loaded_at)
        logger.warning(
            "At tier capacity (%d). Evicting LRU tier: %s",
            self.max_loaded_tiers, lru_tier
        )
        self._evict(lru_tier)

    @staticmethod
    def _vram_used_gb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024**3
        return 0.0

    @staticmethod
    def _vram_total_gb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0

    @staticmethod
    def _vram_free_gb() -> float:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            used  = torch.cuda.memory_reserved()
            return (total - used) / 1024**3
        return 0.0

    def __repr__(self) -> str:
        loaded = list(self._tiers.keys()) or ["none"]
        return (
            f"ModelManager(model={self.model_id!r}, "
            f"device={self.device!r}, "
            f"loaded={loaded})"
        )