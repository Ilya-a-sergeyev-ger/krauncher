# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""KrauncherClient — main entry point for submitting GPU tasks."""

from __future__ import annotations

import functools
import inspect
import logging
import os
import time
from typing import Any, Callable

_logger = logging.getLogger("krauncher")

import httpx

from .analyzer import (
    AnalyzerClient,
    TaskClassification,
    classify_explicit,
    classify_safety_net,
)
from .data_source import DataSource
from .exceptions import KrauncherError
from .models import Runner, TaskHandle, _check_response
from .serializer import serialize_function
from .volume import Volume

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
    gpu_name         KRAUNCHER_GPU_NAME     ""
    gpu_arch         KRAUNCHER_GPU_ARCH     ""
    estimate_only    CAS_ESTIMATE_ONLY      false
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
        gpu_name: str | None = None,
        gpu_arch: str | None = None,
        estimate_only: bool | None = None,
        stream_stderr: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("CAS_API_KEY", "")
        if not self.api_key:
            raise KrauncherError(
                "Missing API key. Pass api_key=... to KrauncherClient(), or set "
                "the CAS_API_KEY environment variable (e.g. in a .env file). "
                "Generate a key at https://krauncher.com → Account → API Keys."
            )
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

        # Default GPU requirements from client config
        self.default_gpu_name = gpu_name or os.environ.get("KRAUNCHER_GPU_NAME", "")
        self.default_gpu_arch = gpu_arch or os.environ.get("KRAUNCHER_GPU_ARCH", "")

        # Estimate-only mode: run analyzer, return classification, skip broker submission.
        if estimate_only is not None:
            self.estimate_only = estimate_only
        else:
            self.estimate_only = os.environ.get("CAS_ESTIMATE_ONLY", "false").lower() in ("1", "true", "yes")

        # Stream stderr from worker to client via relay (worker emits type=stderr,
        # wait() auto-prints to sys.stderr when no on_log is provided).
        if stream_stderr is not None:
            self.stream_stderr = stream_stderr
        else:
            self.stream_stderr = os.environ.get("CAS_STREAM_STDERR", "false").lower() in ("1", "true", "yes")

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
        token = self._get_broker_config().get("analyzer_token")
        # Re-create client if URL changed
        if self._analyzer_client is not None and self._analyzer_client._url == url.rstrip("/"):
            return self._analyzer_client
        self._analyzer_client = AnalyzerClient(
            analyzer_url=url,
            encrypt=self._encrypt_analyzer,
            timeout=self._analyzer_timeout,
            token=token,
        )
        return self._analyzer_client

    async def _resolve_dataset_mb(
        self,
        data: str | None,
        volume: str | None,
    ) -> float | None:
        """Query broker for input dataset size (MB) for CU estimation.

        Only includes input data sources and volumes — output sources are
        excluded because they don't affect training iteration count.
        Returns None if no data/volume specified or on any error (best-effort).
        """
        names = [n for n in (data, volume) if n is not None]
        if not names:
            return None
        try:
            async with httpx.AsyncClient(timeout=5.0) as session:
                resp = await session.post(
                    f"{self.broker_url}/data-sources/sizes",
                    json={"names": names},
                    headers={"X-API-Key": self.api_key},
                )
                if resp.status_code != 200:
                    return None
                sizes = resp.json().get("sizes", {})
                total = sum(v for v in sizes.values() if v is not None)
                return total if total > 0 else None
        except Exception:
            return None

    def task(
        self,
        *,
        vram_gb: int | None = None,
        gpu_arch: str | None = None,
        gpu_name: str | None = None,
        pip: list[str] | None = None,
        timeout: int = 600,
        priority: int = 1,
        data_urls: list[str] | None = None,
        data: str | None = None,
        output: str | None = None,
        volume: str | None = None,
        group_id: str | None = None,
        provider: str | None = None,
        disk_gb: int = 10,
        dataset_size: float | None = None,
        stream_stderr: bool | None = None,
    ) -> Callable:
        """Decorator that marks a function as a remote GPU task.

        The decorated function becomes async — calling it submits the task
        to the broker and returns a :class:`TaskHandle`.

        Args:
            vram_gb: Minimum GPU VRAM in GB.  ``None`` = auto-classify via
                cas-analyzer (or safety net if unavailable).
            gpu_arch: Required GPU architecture (e.g. ``"Ada"``).  ``None`` = use
                client default (from constructor or KRAUNCHER_GPU_ARCH env var).
                Empty string = no filter.
            gpu_name: Required GPU model (case-insensitive substring, e.g. ``"H100"``,
                ``"L4"``).  ``None`` = use client default (from constructor or
                KRAUNCHER_GPU_NAME env var).  Empty string = no filter.
            pip: Pip packages to install in the sandbox before execution.
            timeout: Execution timeout in seconds.
            priority: Task priority (0 = highest, 10 = lowest).
            data_urls: URLs for data bridge downloads into ``/data``.
            data: Registered data source name — broker resolves URLs and
                credentials from the database.  Downloads into ``/data``.
            output: Registered output data source name (is_output=True) —
                broker resolves upload destination.  Task writes to ``/output``.
            volume: Persistent volume name — S3-backed storage synced to
                ``/volume`` before execution and pushed back after.
            group_id: Task group ID for host affinity — tasks with the
                same group_id are routed to the same worker.
            provider: Pin task to a specific provider (e.g. ``"runpod"`` or
                ``"local"``).  ``None`` lets the dispatcher pick the cheapest
                suitable host across all providers.
            disk_gb: Required disk space in GB (default 20).  The broker takes
                the maximum of this value and the auto-resolved size from
                data sources.
            dataset_size: Dataset size in MB for CU estimation.  Overrides
                auto-resolved size from data sources.
        """

        client = self

        def decorator(func: Callable) -> Callable:
            # Serialize at decoration time — fail fast on invalid functions
            code_string, entry_point = serialize_function(func)

            # Extract default values from function signature at decoration time
            _func_defaults: dict[str, Any] = {}
            try:
                sig = inspect.signature(func)
                for name, param in sig.parameters.items():
                    if param.default is not inspect.Parameter.empty:
                        _func_defaults[name] = param.default
            except (ValueError, TypeError):
                pass

            # Cache analyzer result per decorated function — the code_string
            # never changes, so re-analyzing on every call is wasteful and
            # causes timeouts under concurrent load.
            _cached_classification: list[TaskClassification | None] = [None]

            @functools.wraps(func)
            async def wrapper(**kwargs: Any) -> TaskHandle:
                import time as _time
                _submit_start = _time.monotonic()
                # Merge defaults with passed kwargs (passed values take priority)
                merged_kwargs = {**_func_defaults, **kwargs}

                # Classification: call analyzer once, cache for subsequent calls.
                if _cached_classification[0] is not None:
                    classification = _cached_classification[0]
                else:
                    # _analyzer raises KrauncherError if no analyzer configured
                    try:
                        # Query broker for data source sizes to improve CU estimation
                        dataset_mb = dataset_size or await client._resolve_dataset_mb(data, volume)
                        classification = await client._analyzer.classify(
                            code_string, dataset_mb=dataset_mb, kwargs=merged_kwargs,
                        )
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

                if _logger.isEnabledFor(logging.DEBUG):
                    c = classification
                    cu_str = str(c.compute_units)
                    if c.cu_compute is not None:
                        cu_str += f" (compute={c.cu_compute}, io={c.cu_io}"
                        if c.model_download_mb is not None:
                            cu_str += f", model={c.model_download_mb:.0f}MB"
                        if c.dataset_mb is not None:
                            cu_str += f", dataset={c.dataset_mb:.0f}MB"
                        cu_str += ")"
                    parts = [
                        f"tier={c.tier}",
                        f"VRAM={c.min_vram_gb}GB",
                        f"CU={cu_str}",
                        f"method={c.analysis_method}",
                    ]
                    if c.cpu_only:
                        parts.append("cpu_only=True")
                    if c.input_tokens is not None:
                        parts.append(f"input_tokens={c.input_tokens}")
                    if c.seq_len is not None:
                        parts.append(f"seq_len={c.seq_len}")
                    if c.workload_type:
                        parts.append(f"workload={c.workload_type}")
                    if c.model_size_category:
                        parts.append(f"model_size={c.model_size_category}")
                    if c.working_set_category:
                        parts.append(f"working_set={c.working_set_category}")
                    if c.data_per_step:
                        data_str = f"{c.data_per_step_gb:.1f}GB" if c.data_per_step_gb else ""
                        parts.append(f"data/step={c.data_per_step}({data_str})")
                    if c.compute_per_step:
                        comp_str = f"{c.compute_per_step_tflops:.2f}TF" if c.compute_per_step_tflops else ""
                        parts.append(f"compute/step={c.compute_per_step}({comp_str})")
                    if c.resource_profile:
                        rp = c.resource_profile
                        parts.append(
                            f"profile=[ci={rp.get('compute_intensity', 0):.2f},"
                            f"si={rp.get('storage_io_sensitivity', 0):.2f},"
                            f"cu={rp.get('cpu_utilization', 0):.2f},"
                            f"pcie={rp.get('pcie_bandwidth_util', 0):.2f},"
                            f"net={rp.get('network_io_sensitivity', 0):.2f}]"
                        )
                    if c.analyzer_time is not None:
                        parts.append(f"time={c.analyzer_time:.2f}s")
                    _logger.debug("Classification: %s", ", ".join(parts))

                if client.estimate_only:
                    import sys as _sys
                    c = classification
                    _logger.info(
                        "estimate_only=true — skipping broker submission "
                        "(CU=%s, VRAM=%sGB, tier=%s, method=%s, cpu_only=%s)",
                        c.compute_units, c.min_vram_gb, c.tier, c.analysis_method, c.cpu_only,
                    )
                    _sys.exit(0)

                # Priority: decorator param → client default (from env or constructor)
                final_gpu_arch = gpu_arch if gpu_arch is not None else client.default_gpu_arch
                final_gpu_name = gpu_name if gpu_name is not None else client.default_gpu_name

                requirements: dict[str, Any] = {
                    "min_vram_gb": classification.min_vram_gb,
                    "gpu_arch": final_gpu_arch,
                    "gpu_name": final_gpu_name,
                    "disk_gb": disk_gb,
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
                        "args": merged_kwargs,
                        "pip": pip or [],
                    }

                effective_stream_stderr = (
                    stream_stderr if stream_stderr is not None else client.stream_stderr
                )

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
                        "stream_stderr": effective_stream_stderr,
                    },
                }

                if group_id is not None:
                    body["group_id"] = group_id
                if data is not None:
                    body["data"] = data
                if output is not None:
                    body["output"] = output
                if volume is not None:
                    body["volume"] = volume

                body["classification"] = classification.to_dict()

                async def _post_task() -> str:
                    async with httpx.AsyncClient(timeout=30.0) as session:
                        resp = await session.post(
                            f"{client.broker_url}/tasks",
                            json=body,
                            headers={"X-API-Key": client.api_key},
                        )
                        _check_response(resp)
                        return resp.json()["task_id"]

                task_id = await _post_task()
                return TaskHandle(
                    task_id=task_id,
                    client=client,
                    ek_priv=ek_priv,
                    plaintext_code=code_string if client.encrypt else None,
                    plaintext_args=kwargs if client.encrypt else None,
                    classification=classification,
                    submit_start=_submit_start,
                    resubmit=_post_task,
                    stream_stderr=effective_stream_stderr,
                )

            # Store metadata for introspection
            wrapper._krauncher_code = code_string
            wrapper._krauncher_entry_point = entry_point
            wrapper._krauncher_pip = pip or []
            wrapper._krauncher_provider = provider

            return wrapper

        return decorator

    def data_source(
        self,
        name: str,
        urls: list[str] | None = None,
        size_gb: float = 0,
        description: str | None = None,
        is_output: bool = False,
    ) -> DataSource:
        """Create or get a registered data source.

        If *urls* is provided, registers a new data source on the broker.
        Otherwise returns a handle to an existing source (for inspection
        or deletion).

        Args:
            name: Unique name for the data source.
            urls: S3 or HTTP URLs to register.
            size_gb: Declared data size in GB.
            description: Optional description.
            is_output: If ``True``, this source is used for uploading task results.

        Returns:
            A :class:`DataSource` handle.
        """
        return DataSource(self, name, urls, size_gb, description, is_output)

    def volume(self, name: str, size_gb: int = 5) -> Volume:
        """Create or get a persistent volume.

        Ensures the volume exists on the broker (creates if missing).

        Args:
            name: Volume name.
            size_gb: Quota in GB (used only on creation).

        Returns:
            A :class:`Volume` handle with ``upload()``, ``download()``,
            ``ls()``, and ``delete()`` methods.
        """
        return Volume(self, name, size_gb)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Fetch the full task record by id (same payload as ``GET /tasks/{id}``).

        Mirrors what the user sees on the task detail page in the web UI:
        status, timing breakdown, classification, costs, GPU and worker specs,
        and the result. Intended for programmatic inspection of a task — e.g.
        as feedback to an LLM that authored the user code.
        """
        async with httpx.AsyncClient(timeout=30.0) as session:
            resp = await session.get(
                f"{self.broker_url}/tasks/{task_id}",
                headers={"X-API-Key": self.api_key},
            )
            _check_response(resp)
            return resp.json()

    async def get_task_report(self, task_id: str) -> dict[str, Any]:
        """Fetch the extended analytics report for a finished task.

        Returns peak/avg GPU utilization, peak VRAM, the actual GPU's hardware
        specs, and an estimated time/cost comparison across all known GPUs at
        the worker's measured host capabilities. Companion to :meth:`get_task`
        — the basic call returns what the broker already records, this one
        bundles the additional data needed for performance analysis without
        requiring the LLM to re-derive it.

        Returns the merged dict ``{**get_task(...), "report": {...}}`` so the
        caller has the full picture in a single object.
        """
        async with httpx.AsyncClient(timeout=30.0) as session:
            task_resp = await session.get(
                f"{self.broker_url}/tasks/{task_id}",
                headers={"X-API-Key": self.api_key},
            )
            _check_response(task_resp)
            task = task_resp.json()

            report_resp = await session.get(
                f"{self.broker_url}/tasks/{task_id}/report",
                headers={"X-API-Key": self.api_key},
            )
            _check_response(report_resp)
            report = report_resp.json()

        return {**task, "report": report}

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
