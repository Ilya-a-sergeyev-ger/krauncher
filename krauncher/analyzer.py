# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

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
    cpu_only: bool = False           # task makes no use of the GPU (analyzer flag)
    compute_units: float | None = None
    cu_compute: float | None = None           # compute phase CU (GPU + DataLoader pipeline)
    cu_io: float | None = None                # IO phase CU (model/dataset download + pip)
    cu_setup: float | None = None             # setup phase CU (torch import + CUDA init)
    seq_len: int | None = None                # decode/generation sequence length
    input_tokens: int | None = None           # prompt length P (llm_inference prefill)
    predicted_sec: float | None = None        # reference-time forecast (t_setup + t_io_ref + t_compute_ref)
    model_download_mb: float | None = None    # estimated model download size (MB)
    dataset_mb: float | None = None           # dataset size (MB)
    duration_confidence: float | None = None
    workload_type: str | None = None       # "llm_inference" | "ai_training" | "cv_training" | ...
    model_size_category: str | None = None    # "small" | "medium" | "large"
    working_set_category: str | None = None   # "small" | "medium" | "large"
    data_per_step_gb: float | None = None     # bytes moved through HBM per step (GB)
    data_per_step: str | None = None          # "small" | "medium" | "large"
    compute_per_step_tflops: float | None = None  # FLOPS per step (TFLOP)
    compute_per_step: str | None = None       # "small" | "medium" | "large"
    resource_profile: dict | None = None      # 8-dim host resource appetite (for EHP)
    epochs_bucket: str | None = None          # "one" | "few" | "many"
    samples_bucket: str | None = None         # "tiny" | "small" | "medium" | "large"
    cu_findings: list[str] = field(default_factory=list)  # training loop breakdown from analyzer
    analyzer_time: float | None = None              # analyzer round-trip time in seconds
    analyzer_job_id: str | None = None              # analyzer's job id (correlation key for results)
    # Universal pass-through: any analyzer duration_estimate field not mapped to a
    # typed attribute above (e.g. cu_prefill/cu_decode and future phase metadata)
    # is carried here and forwarded as-is to the broker payload — so new analyzer
    # debug fields need no client change.
    extra_debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "min_vram_gb": self.min_vram_gb,
            "tier": self.tier,
            "confidence": self.confidence,
            "analysis_method": self.analysis_method,
            "cpu_only": self.cpu_only,
        }
        if self.compute_units is not None:
            d["compute_units"] = self.compute_units
        if self.cu_compute is not None:
            d["cu_compute"] = self.cu_compute
        if self.cu_io is not None:
            d["cu_io"] = self.cu_io
        if self.cu_setup is not None:
            d["cu_setup"] = self.cu_setup
        if self.seq_len is not None:
            d["seq_len"] = self.seq_len
        if self.input_tokens is not None:
            d["input_tokens"] = self.input_tokens
        if self.predicted_sec is not None:
            d["predicted_sec"] = self.predicted_sec
        if self.model_download_mb is not None:
            d["model_download_mb"] = self.model_download_mb
        if self.dataset_mb is not None:
            d["dataset_mb"] = self.dataset_mb
        if self.duration_confidence is not None:
            d["duration_confidence"] = self.duration_confidence
        if self.workload_type is not None:
            d["workload_type"] = self.workload_type
        if self.model_size_category is not None:
            d["model_size_category"] = self.model_size_category
        if self.working_set_category is not None:
            d["working_set_category"] = self.working_set_category
        if self.data_per_step is not None:
            d["data_per_step"] = self.data_per_step
        if self.compute_per_step is not None:
            d["compute_per_step"] = self.compute_per_step
        if self.data_per_step_gb is not None:
            d["data_per_step_gb"] = self.data_per_step_gb
        if self.compute_per_step_tflops is not None:
            d["compute_per_step_tflops"] = self.compute_per_step_tflops
        if self.resource_profile is not None:
            d["resource_profile"] = self.resource_profile
        if self.epochs_bucket is not None:
            d["epochs_bucket"] = self.epochs_bucket
        if self.samples_bucket is not None:
            d["samples_bucket"] = self.samples_bucket
        if self.analyzer_job_id is not None:
            d["analyzer_job_id"] = self.analyzer_job_id
        # Universal pass-through of unmapped analyzer fields (debug/phase metadata).
        for k, v in self.extra_debug.items():
            if v is not None and k not in d:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# Tier mapping
# ---------------------------------------------------------------------------

# VRAM safety headroom applied to every requirement before GPU selection, so a
# task is never scheduled onto a card it only just fits (real cards deliver
# below their nominal VRAM, and the estimate itself carries uncertainty). Same
# factor on both the explicit pin and the analyzer's auto-classified estimate.
_VRAM_HEADROOM = 1.1


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
    """Level 1: user explicitly set vram_gb. Add safety headroom."""
    effective = math.ceil(vram_gb * _VRAM_HEADROOM)
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
        token: str | None = None,
        user_id: str | None = None,
        llm_backend: str | None = None,
    ) -> None:
        self._url = analyzer_url.rstrip("/")
        self._encrypt = encrypt
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._user_id = user_id
        self._llm_backend = llm_backend
        self._analyzer_pubkey: bytes | None = None
        self._headers: dict[str, str] = (
            {"X-Analyzer-Token": token} if token else {}
        )

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
        kwargs: dict | None = None,
    ) -> TaskClassification:
        """Call cas-analyzer and return classification.

        On decryption error, invalidates cached pubkey and retries once.
        Raises on any other error (caller handles fallback).
        """
        return await self._classify_inner(code, dataset_mb, kwargs=kwargs, retry=True)

    async def _classify_inner(
        self,
        code: str,
        dataset_mb: int | None,
        retry: bool,
        kwargs: dict | None = None,
    ) -> TaskClassification:
        import asyncio
        import logging as _log
        import time as _time

        _logger = _log.getLogger("krauncher")
        t0 = _time.monotonic()

        async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as session:
            # Build request body
            body: dict = {
                "source": "api",
                "user_id": self._user_id,
                "llm_backend": self._llm_backend,
            }
            if dataset_mb is not None:
                body["dataset_mb"] = dataset_mb
            if kwargs:
                # Filter to JSON-safe scalar values only
                safe_kwargs = {
                    k: v for k, v in kwargs.items()
                    if isinstance(v, (int, float, bool, str))
                }
                if safe_kwargs:
                    body["kwargs"] = safe_kwargs

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
                return await self._classify_inner(code, dataset_mb, kwargs=kwargs, retry=False)
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
                    elapsed = _time.monotonic() - t0
                    result = self._parse_result(data["result"])
                    result.analyzer_time = elapsed
                    result.analyzer_job_id = job_id
                    if result.cu_findings:
                        for finding in result.cu_findings:
                            _logger.debug("  CU: %s", finding)
                    return result
                elif data["status"] == "failed":
                    raise RuntimeError(f"Analyzer failed: {data.get('error', 'unknown')}")
                elif asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError(f"Analyzer timed out after {self._timeout}s")

    @staticmethod
    def _parse_result(result: dict) -> TaskClassification:
        """Parse cas-analyzer result into TaskClassification."""
        hw = result.get("min_hardware", {})
        dur = result.get("duration_estimate")

        raw_vram_gb = hw.get("min_vram_gb", 24)
        # Same safety headroom as the explicit path (classify_explicit); 0 = CPU
        # task stays 0.
        min_vram_gb = math.ceil(raw_vram_gb * _VRAM_HEADROOM)
        method = hw.get("analysis_method", "ast")
        confidence = hw.get("confidence", 0.6)

        cu = None
        cu_compute = None
        cu_io = None
        cu_setup = None
        predicted_sec = None
        model_download_mb = None
        dataset_mb_val = None
        dur_conf = None
        cu_findings: list[str] = []
        epochs_bkt = None
        samples_bkt = None
        seq_len_val = None
        input_tokens_val = None
        if dur:
            cu = dur.get("compute_units")
            cu_compute = dur.get("cu_compute")
            cu_io = dur.get("cu_io")
            cu_setup = dur.get("cu_setup")
            predicted_sec = dur.get("predicted_sec")
            model_download_mb = dur.get("model_download_mb")
            dataset_mb_val = dur.get("dataset_mb")
            dur_conf = dur.get("confidence")
            cu_findings = dur.get("findings") or []
            epochs_bkt = dur.get("epochs_bucket")
            samples_bkt = dur.get("samples_bucket")
            seq_len_val = dur.get("seq_len")
            input_tokens_val = dur.get("input_tokens")

        # Agnostic pass-through: forward every analyzer field. Keys already emitted
        # as typed attributes are dropped by to_dict()'s `k not in d` guard, so new
        # analyzer signals (dataloader_num_workers, ...) need no client change.
        extra_debug = {k: v for k, v in (dur or {}).items() if v is not None}
        extra_debug.update({k: v for k, v in hw.items() if v is not None})

        return TaskClassification(
            min_vram_gb=min_vram_gb,
            tier=_vram_to_tier(min_vram_gb),
            confidence=confidence,
            analysis_method=method,
            cpu_only=bool(hw.get("cpu_only", False)),
            compute_units=cu,
            cu_compute=cu_compute,
            cu_io=cu_io,
            cu_setup=cu_setup,
            seq_len=seq_len_val,
            input_tokens=input_tokens_val,
            predicted_sec=predicted_sec,
            model_download_mb=model_download_mb,
            dataset_mb=dataset_mb_val,
            duration_confidence=dur_conf,
            workload_type=hw.get("workload_type"),
            model_size_category=hw.get("model_size_category"),
            working_set_category=hw.get("working_set_category"),
            data_per_step_gb=hw.get("data_per_step_gb"),
            data_per_step=hw.get("data_per_step"),
            compute_per_step_tflops=hw.get("compute_per_step_tflops"),
            compute_per_step=hw.get("compute_per_step"),
            resource_profile=hw.get("resource_profile"),
            epochs_bucket=epochs_bkt,
            samples_bucket=samples_bkt,
            cu_findings=cu_findings,
            extra_debug=extra_debug,
        )
