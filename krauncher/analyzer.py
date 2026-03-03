"""Client-side task classification via cas-analyzer.

Three classification levels:
  Level 1 — Explicit: user provides vram_gb directly
  Level 2 — Analyzer: cas-analyzer AST/LLM analysis (E2E encrypted)
  Level 3 — Safety Net: fallback 24GB / light / confidence=0.5
"""

from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass, field

import httpx

from .crypto import generate_keypair, derive_shared_secret, encrypt

logger = logging.getLogger("krauncher.analyzer")


# ---------------------------------------------------------------------------
# TaskClassification dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskClassification:
    min_vram_gb: int
    tier: str                        # "no_gpu" | "light" | "heavy"
    confidence: float                # 0.0–1.0
    analysis_method: str             # "explicit" | "ast" | "ast+llm" | "safety_net"
    performance_table: list[dict] = field(default_factory=list)
    compute_units: float | None = None
    duration_confidence: float | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "min_vram_gb": self.min_vram_gb,
            "tier": self.tier,
            "confidence": self.confidence,
            "analysis_method": self.analysis_method,
            "performance_table": self.performance_table,
        }
        if self.compute_units is not None:
            d["compute_units"] = self.compute_units
        if self.duration_confidence is not None:
            d["duration_confidence"] = self.duration_confidence
        return d


# ---------------------------------------------------------------------------
# Tier mapping
# ---------------------------------------------------------------------------

def _vram_to_tier(vram_gb: int) -> str:
    if vram_gb == 0:
        return "no_gpu"
    elif vram_gb <= 24:
        return "light"
    else:
        return "heavy"


# ---------------------------------------------------------------------------
# Level 1: Explicit
# ---------------------------------------------------------------------------

def classify_explicit(vram_gb: int) -> TaskClassification:
    """Level 1: user explicitly set vram_gb. Add 10% headroom."""
    effective = math.ceil(vram_gb * 1.1)
    return TaskClassification(
        min_vram_gb=effective,
        tier=_vram_to_tier(effective),
        confidence=1.0,
        analysis_method="explicit",
    )


# ---------------------------------------------------------------------------
# Level 3: Safety Net
# ---------------------------------------------------------------------------

def classify_safety_net() -> TaskClassification:
    """Level 3: fallback when analyzer is unavailable or fails."""
    return TaskClassification(
        min_vram_gb=6,
        tier="light",
        confidence=0.5,
        analysis_method="safety_net",
    )


# ---------------------------------------------------------------------------
# Level 2: AnalyzerClient (E2E encrypted)
# ---------------------------------------------------------------------------

class AnalyzerClient:
    """Async client for cas-analyzer with optional E2E encryption."""

    def __init__(
        self,
        analyzer_url: str,
        encrypt: bool = True,
        timeout: float = 10.0,
        poll_interval: float = 0.5,
    ) -> None:
        self._url = analyzer_url.rstrip("/")
        self._encrypt = encrypt
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._analyzer_pubkey: bytes | None = None

    async def _fetch_pubkey(self, session: httpx.AsyncClient) -> bytes:
        """GET /pubkey — fetch and cache the analyzer's public key."""
        if self._analyzer_pubkey is not None:
            return self._analyzer_pubkey
        resp = await session.get(f"{self._url}/pubkey")
        resp.raise_for_status()
        pub_b64 = resp.json()["public_key"]
        self._analyzer_pubkey = base64.urlsafe_b64decode(pub_b64 + "==")
        return self._analyzer_pubkey

    async def classify(
        self,
        code: str,
        dataset_mb: int | None = None,
    ) -> TaskClassification:
        """Call cas-analyzer and return classification.

        On decryption error, invalidates cached pubkey and retries once.
        Raises on any other error (caller handles fallback).
        """
        return await self._classify_inner(code, dataset_mb, retry=True)

    async def _classify_inner(
        self,
        code: str,
        dataset_mb: int | None,
        retry: bool,
    ) -> TaskClassification:
        import asyncio

        async with httpx.AsyncClient(timeout=self._timeout) as session:
            # Build request body
            body: dict = {}
            if dataset_mb is not None:
                body["dataset_mb"] = dataset_mb

            if self._encrypt:
                pub_bytes = await self._fetch_pubkey(session)
                ek_priv, ek_pub_bytes = generate_keypair()
                shared_secret = derive_shared_secret(ek_priv, pub_bytes)
                encrypted_code = encrypt(shared_secret, code.encode("utf-8"))
                ek_pub_b64 = base64.urlsafe_b64encode(ek_pub_bytes).decode().rstrip("=")
                body["encrypted_code"] = encrypted_code
                body["client_public_key"] = ek_pub_b64
            else:
                body["code"] = code

            # POST /analyze
            resp = await session.post(f"{self._url}/analyze", json=body)
            if resp.status_code == 400 and retry and self._encrypt:
                # Possible key rotation — clear cache and retry once
                self._analyzer_pubkey = None
                return await self._classify_inner(code, dataset_mb, retry=False)
            resp.raise_for_status()
            job_id = resp.json()["job_id"]

            # Poll GET /jobs/{job_id}
            deadline = asyncio.get_event_loop().time() + self._timeout
            while True:
                await asyncio.sleep(self._poll_interval)
                poll_resp = await session.get(f"{self._url}/jobs/{job_id}")
                poll_resp.raise_for_status()
                data = poll_resp.json()

                if data["status"] == "done":
                    return self._parse_result(data["result"])
                elif data["status"] == "failed":
                    raise RuntimeError(f"Analyzer failed: {data.get('error', 'unknown')}")
                elif asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError(f"Analyzer timed out after {self._timeout}s")

    @staticmethod
    def _parse_result(result: dict) -> TaskClassification:
        """Parse cas-analyzer result into TaskClassification."""
        hw = result.get("min_hardware", {})
        dur = result.get("duration_estimate")
        perf = result.get("performance_table", [])

        min_vram_gb = hw.get("min_vram_gb", 24)
        method = hw.get("analysis_method", "ast")
        confidence = hw.get("confidence", 0.6)

        cu = None
        dur_conf = None
        if dur:
            cu = dur.get("compute_units")
            dur_conf = dur.get("confidence")

        return TaskClassification(
            min_vram_gb=min_vram_gb,
            tier=_vram_to_tier(min_vram_gb),
            confidence=confidence,
            analysis_method=method,
            performance_table=perf,
            compute_units=cu,
            duration_confidence=dur_conf,
        )
