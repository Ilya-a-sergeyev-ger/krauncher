"""KrauncherClient — main entry point for submitting GPU tasks."""

from __future__ import annotations

import functools
import os
from typing import Any, Callable

import httpx

from .analyzer import (
    AnalyzerClient,
    TaskClassification,
    classify_explicit,
    classify_safety_net,
)
from .exceptions import KrauncherError
from .models import Runner, TaskHandle, _check_response
from .serializer import serialize_function

# Sentinel to distinguish "not passed" from explicit None
_UNSET: Any = object()


class KrauncherClient:
    """Client for submitting tasks to the CaS broker.

    All parameters can be set via environment variables (or ``.env`` file in CWD).
    Explicit constructor arguments always take priority.

    ================ ====================== ==========================================
    Parameter        Env var                Default
    ================ ====================== ==========================================
    api_key          CAS_API_KEY            (required)
    broker_url       CAS_BROKER_URL         https://krauncher.com
    encrypt          CAS_ENCRYPT            true
    analyzer_url     CAS_ANALYZER_URL       None (no analyzer → safety net)
    encrypt_analyzer CAS_ENCRYPT_ANALYZER   true
    analyzer_timeout CAS_ANALYZER_TIMEOUT   10.0
    ================ ====================== ==========================================

    Usage::

        # All config from .env:
        client = KrauncherClient()

        # Or explicit:
        client = KrauncherClient(api_key="cas_...", broker_url="http://...")

        @client.task(timeout=3600)
        def train(data):
            import torch
            return {"loss": 0.01}

        handle = await train(data={"epochs": 5})
        result = await handle
    """

    def __init__(
        self,
        api_key: str | None = None,
        broker_url: str | None = None,
        encrypt: bool | None = None,
        analyzer_url: Any = _UNSET,
        encrypt_analyzer: bool | None = None,
        analyzer_timeout: float | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("CAS_API_KEY", "")
        self.broker_url = (broker_url or os.environ.get("CAS_BROKER_URL", "https://krauncher.com")).rstrip("/")

        if encrypt is not None:
            self.encrypt = encrypt
        else:
            self.encrypt = os.environ.get("CAS_ENCRYPT", "true").lower() not in ("0", "false", "no")

        if analyzer_url is not _UNSET:
            self._analyzer_url = analyzer_url
        else:
            self._analyzer_url = os.environ.get("CAS_ANALYZER_URL") or None

        if encrypt_analyzer is not None:
            self._encrypt_analyzer = encrypt_analyzer
        else:
            self._encrypt_analyzer = os.environ.get("CAS_ENCRYPT_ANALYZER", "true").lower() not in ("0", "false", "no")

        self._analyzer_timeout = analyzer_timeout or float(os.environ.get("CAS_ANALYZER_TIMEOUT", "10.0"))
        self._analyzer_client: AnalyzerClient | None = None

    @property
    def _analyzer(self) -> AnalyzerClient | None:
        """Lazy-init AnalyzerClient."""
        if self._analyzer_url and self._analyzer_client is None:
            self._analyzer_client = AnalyzerClient(
                analyzer_url=self._analyzer_url,
                encrypt=self._encrypt_analyzer,
                timeout=self._analyzer_timeout,
            )
        return self._analyzer_client

    def task(
        self,
        *,
        vram_gb: int | None = None,
        gpu_arch: str = "Ampere+",
        pip: list[str] | None = None,
        timeout: int = 600,
        priority: int = 1,
        data_urls: list[str] | None = None,
        group_id: str | None = None,
        provider: str | None = None,
    ) -> Callable:
        """Decorator that marks a function as a remote GPU task.

        The decorated function becomes async — calling it submits the task
        to the broker and returns a :class:`TaskHandle`.

        Args:
            vram_gb: Minimum GPU VRAM in GB.  ``None`` = auto-classify via
                cas-analyzer (or safety net if unavailable).
            gpu_arch: Required GPU architecture (e.g. ``"Ampere+"``).
            pip: Pip packages to install in the sandbox before execution.
            timeout: Execution timeout in seconds.
            priority: Task priority (0 = highest, 10 = lowest).
            data_urls: URLs for data bridge downloads into ``/data``.
            group_id: Task group ID for host affinity — tasks with the
                same group_id are routed to the same worker.
            provider: Pin task to a specific provider (e.g. ``"runpod"`` or
                ``"local"``).  ``None`` lets the dispatcher pick the cheapest
                suitable host across all providers.
        """

        client = self

        def decorator(func: Callable) -> Callable:
            # Serialize at decoration time — fail fast on invalid functions
            code_string, entry_point = serialize_function(func)

            # Cache analyzer result per decorated function — the code_string
            # never changes, so re-analyzing on every call is wasteful and
            # causes timeouts under concurrent load.
            _cached_classification: list[TaskClassification | None] = [None]

            @functools.wraps(func)
            async def wrapper(**kwargs: Any) -> TaskHandle:
                # Classification: call analyzer once, cache for subsequent calls.
                if _cached_classification[0] is not None:
                    classification = _cached_classification[0]
                elif client._analyzer:
                    try:
                        classification = await client._analyzer.classify(code_string)
                    except Exception as exc:
                        import logging as _log
                        _log.getLogger("krauncher").warning("Analyzer failed, using safety net: %s", exc)
                        classification = classify_safety_net()
                    _cached_classification[0] = classification
                else:
                    classification = classify_safety_net()
                    _cached_classification[0] = classification

                if vram_gb is not None:
                    # Level 1 override: keep analyzer's compute_units/duration/perf_table,
                    # but force vram_gb (with 10% headroom) and recalculate tier.
                    # Copy first — cached classification is shared across calls.
                    import dataclasses
                    classification = dataclasses.replace(classification)
                    explicit = classify_explicit(vram_gb)
                    classification.min_vram_gb = explicit.min_vram_gb
                    classification.tier = explicit.tier
                    classification.confidence = explicit.confidence
                    classification.analysis_method = explicit.analysis_method

                requirements: dict[str, Any] = {
                    "min_vram_gb": classification.min_vram_gb,
                    "gpu_arch": gpu_arch,
                }
                if provider is not None:
                    requirements["provider_name"] = provider

                # E2E encryption: generate ephemeral keypair, withhold plaintext code
                ek_priv = None
                if client.encrypt:
                    import base64
                    from .crypto import generate_keypair
                    ek_priv, ek_pub_bytes = generate_keypair()
                    ek_pub_b64 = base64.urlsafe_b64encode(ek_pub_bytes).decode().rstrip("=")
                    payload_body: dict[str, Any] = {
                        "code_string": "",
                        "entry_point": entry_point,
                        "args": {},
                        "pip": pip or [],
                        "encryption_key": ek_pub_b64,
                    }
                else:
                    payload_body = {
                        "code_string": code_string,
                        "entry_point": entry_point,
                        "args": kwargs,
                        "pip": pip or [],
                    }

                body: dict[str, Any] = {
                    "priority": priority,
                    "requirements": requirements,
                    "payload": payload_body,
                    "data_bridge": {
                        "download_urls": data_urls or [],
                        "mount_path": "/data",
                    },
                    "limits": {
                        "timeout_sec": timeout,
                    },
                }

                if group_id is not None:
                    body["group_id"] = group_id

                body["classification"] = classification.to_dict()

                async with httpx.AsyncClient(timeout=30.0) as session:
                    resp = await session.post(
                        f"{client.broker_url}/tasks",
                        json=body,
                        headers={"X-API-Key": client.api_key},
                    )
                    _check_response(resp)
                    task_id = resp.json()["task_id"]
                    return TaskHandle(
                        task_id=task_id,
                        client=client,
                        ek_priv=ek_priv,
                        plaintext_code=code_string if client.encrypt else None,
                        plaintext_args=kwargs if client.encrypt else None,
                        classification=classification,
                    )

            # Store metadata for introspection
            wrapper._krauncher_code = code_string
            wrapper._krauncher_entry_point = entry_point
            wrapper._krauncher_pip = pip or []
            wrapper._krauncher_provider = provider

            return wrapper

        return decorator

    async def list_runners(self, *, print_table: bool = True) -> list[Runner]:
        """Fetch available compute runners from the broker fleet.

        Calls ``GET /admin/fleet`` and returns a list of :class:`Runner`
        objects grouped by provider (local first, then external providers
        sorted alphabetically).

        Args:
            print_table: When ``True`` (default), also prints a formatted
                table to stdout — useful in notebooks and interactive shells.

        Returns:
            List of :class:`Runner` objects representing current fleet state.

        Example::

            runners = await client.list_runners()
            # Pick the provider you want:
            runpod_runners = [r for r in runners if r.provider == "runpod"]

            @client.task(vram_gb=24, provider="runpod")
            def train(data): ...
        """
        async with httpx.AsyncClient(timeout=10.0) as session:
            resp = await session.get(
                f"{self.broker_url}/admin/fleet",
                headers={"X-API-Key": self.api_key},
            )
            _check_response(resp)
            data = resp.json()

        # Build worker_id lookup: host_id → worker_id
        workers_by_host: dict[str, str] = {
            w["host_id"]: w["worker_id"]
            for w in data.get("workers", [])
            if w.get("host_id") and w.get("worker_id")
        }

        runners: list[Runner] = []
        for h in data.get("hosts", []):
            runners.append(Runner(
                provider=h.get("provider_name", "unknown"),
                host_id=h.get("host_id", ""),
                gpu_model=h.get("gpu_model", "unknown"),
                gpu_count=h.get("gpu_count", 1),
                vram_gb=h.get("vram_gb", 0),
                gpu_arch=h.get("gpu_arch", "unknown"),
                price_per_hour_usd=h.get("price_per_hour_usd", 0.0),
                status=h.get("status", "unknown"),
                spot=h.get("spot", False),
                region=h.get("region", ""),
                worker_id=workers_by_host.get(h.get("host_id", "")),
            ))

        # Sort: local first, then alphabetically by provider, then by status
        _provider_order = {"local": 0, "mock": 1}
        runners.sort(key=lambda r: (
            _provider_order.get(r.provider, 99),
            r.provider,
            r.status,
            r.host_id,
        ))

        if print_table:
            _print_runners_table(runners)

        return runners


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_runners_table(runners: list[Runner]) -> None:
    """Print a formatted runners table grouped by provider."""
    from .models import _STATUS_SYMBOL  # noqa: PLC0415

    if not runners:
        print("No runners available.")
        return

    cols = ("", "PROVIDER", "GPU", "VRAM", "ARCH", "PRICE/HR", "STATUS", "HOST ID")
    widths = [2, 8, 20, 5, 8, 9, 13, 24]

    sep = "  ".join("-" * w for w in widths)
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))

    print(header)
    print(sep)

    current_provider = None
    for r in runners:
        if r.provider != current_provider:
            if current_provider is not None:
                print()
            current_provider = r.provider

        symbol = _STATUS_SYMBOL.get(r.status, "?")
        price = f"${r.price_per_hour_usd:.2f}" if r.price_per_hour_usd else "free"
        spot_marker = "*" if r.spot else ""
        row = (
            symbol,
            r.provider,
            r.gpu_model[:20],
            f"{r.vram_gb}GB",
            r.gpu_arch[:8],
            f"{price}{spot_marker}",
            r.status,
            r.host_id[:24],
        )
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
