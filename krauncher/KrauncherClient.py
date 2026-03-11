"""KrauncherClient — main entry point for submitting GPU tasks."""

from __future__ import annotations

import functools
import os
import time
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

# Default TTL for broker config cache (seconds)
_CONFIG_CACHE_TTL: float = 900.0  # 15 minutes


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
    encrypt_analyzer CAS_ENCRYPT_ANALYZER   true
    analyzer_timeout CAS_ANALYZER_TIMEOUT   10.0
    ================ ====================== ==========================================

    Analyzer URL is resolved from the broker (``GET /v1/me → analyzer_url``).
    Configure analyzer endpoints in the admin panel.

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

        # analyzer_url is resolved exclusively from the broker (/v1/me).
        # The constructor parameter is kept only for tests / edge cases.
        self._analyzer_url_override = analyzer_url if analyzer_url is not _UNSET else None

        if encrypt_analyzer is not None:
            self._encrypt_analyzer = encrypt_analyzer
        else:
            self._encrypt_analyzer = os.environ.get("CAS_ENCRYPT_ANALYZER", "true").lower() not in ("0", "false", "no")

        self._analyzer_timeout = analyzer_timeout or float(os.environ.get("CAS_ANALYZER_TIMEOUT", "10.0"))
        self._analyzer_client: AnalyzerClient | None = None

        # Broker config cache (populated by _fetch_broker_config)
        self._config_cache: dict[str, Any] | None = None
        self._config_cache_ts: float = 0.0

    def _get_analyzer_url(self) -> str:
        """Return the analyzer URL from broker config.

        Raises KrauncherError if no analyzer is configured.
        """
        if self._analyzer_url_override is not None:
            return self._analyzer_url_override
        config = self._get_broker_config()
        url = config.get("analyzer_url")
        if not url:
            raise KrauncherError(
                "No analyzer endpoint configured on the broker. "
                "An admin must add an active analyzer in the admin panel "
                "(Admin → Resources → Analyzers)."
            )
        return url

    def _get_broker_config(self) -> dict[str, Any]:
        """Return cached broker config, refreshing if TTL expired.

        Raises KrauncherError if the broker is unreachable and no cached
        config is available.
        """
        import logging as _log
        _logger = _log.getLogger("krauncher")

        now = time.monotonic()
        if self._config_cache is not None and (now - self._config_cache_ts) < _CONFIG_CACHE_TTL:
            return self._config_cache
        try:
            import httpx as _httpx
            with _httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self.broker_url}/v1/me",
                    headers={"X-API-Key": self.api_key},
                )
                if resp.status_code == 200:
                    self._config_cache = resp.json()
                    self._config_cache_ts = now
                    return self._config_cache
                _logger.warning("Broker returned %d for GET /v1/me", resp.status_code)
        except Exception as exc:
            _logger.warning("Cannot reach broker at %s: %s", self.broker_url, exc)

        if self._config_cache is not None:
            return self._config_cache  # stale cache on transient failure

        raise KrauncherError(
            f"Cannot reach broker at {self.broker_url}/v1/me — "
            "check broker_url and api_key."
        )

    @property
    def _analyzer(self) -> AnalyzerClient:
        """Lazy-init AnalyzerClient using broker-provided URL.

        Raises KrauncherError if no analyzer is available.
        """
        url = self._get_analyzer_url()  # raises on missing
        # Re-create client if URL changed
        if self._analyzer_client is not None and self._analyzer_client._url == url.rstrip("/"):
            return self._analyzer_client
        self._analyzer_client = AnalyzerClient(
            analyzer_url=url,
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
                else:
                    # _analyzer raises KrauncherError if no analyzer configured
                    try:
                        classification = await client._analyzer.classify(code_string)
                    except KrauncherError:
                        raise
                    except Exception as exc:
                        raise KrauncherError(
                            f"Analyzer failed and CU estimation is unavailable: {exc}"
                        ) from exc
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
