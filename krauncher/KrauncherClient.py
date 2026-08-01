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

from . import _inflight
from .analyzer import (
    AnalyzerClient,
    TaskClassification,
    classify_explicit,
    classify_safety_net,
)
from .credentials import collect_credentials
from .data_source import DataSource
from .exceptions import KrauncherError, ValueTransferError
from .models import Runner, TaskGroup, TaskHandle, _check_response
from .serializer import serialize_function


class _EstimateStub:
    """Permissive placeholder returned by estimate-only handles.

    Any attribute/item access, call or arithmetic yields another stub; formats
    as 0 for numeric specs. Lets result-printing script code after a skipped
    submission run through unchanged in CAS_ESTIMATE_ONLY dry runs.
    """

    # Happy-path values for status-like attributes, so script guards
    # (`if r.status != "completed": abort`) take the success branch.
    _HAPPY: dict = {"status": "completed", "exit_code": 0, "success": True}

    def __getattr__(self, _name: str) -> Any:
        if _name in _EstimateStub._HAPPY:
            return _EstimateStub._HAPPY[_name]
        return self

    def __getitem__(self, _key: Any) -> "_EstimateStub":
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> "_EstimateStub":
        return self

    def __format__(self, spec: str) -> str:
        try:
            return format(0.0, spec)
        except (ValueError, TypeError):
            return "estimate-only"

    def __str__(self) -> str:
        return "estimate-only"

    __repr__ = __str__

    def __float__(self) -> float:
        return 0.0

    def __int__(self) -> int:
        return 0

    def _self(self, *args: Any, **kwargs: Any) -> "_EstimateStub":
        return self

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _self
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = _self

    def _false(self, _other: Any) -> bool:
        return False

    __lt__ = __gt__ = __le__ = __ge__ = _false


class _EstimateOnlyHandle:
    """Stand-in for TaskHandle when estimate_only skips broker submission."""

    task_id = "estimate-only"

    def __init__(self, classification: TaskClassification | None = None):
        self.classification = classification

    async def wait(self, *args: Any, **kwargs: Any) -> _EstimateStub:
        return _EstimateStub()

    async def result(self, *args: Any, **kwargs: Any) -> _EstimateStub:
        return _EstimateStub()

    def __await__(self) -> Any:
        return self.wait().__await__()

    def __getattr__(self, _name: str) -> Any:
        if _name in _EstimateStub._HAPPY:
            return _EstimateStub._HAPPY[_name]
        return _EstimateStub()

# Sentinel to distinguish "not passed" from explicit None
_UNSET: Any = object()

# Default TTL for broker config cache (seconds)
_CONFIG_CACHE_TTL: float = 900.0  # 15 minutes


class KrauncherClient:
    """Client for submitting tasks to the CaS broker.

    All parameters can be set via environment variables (or ``.env`` file in CWD).
    Explicit constructor arguments always take priority.

    ================== ====================== ========================================
    Parameter          Env var                Default
    ================== ====================== ========================================
    api_key            CAS_API_KEY            (required)
    broker_url         CAS_BROKER_URL         https://krauncher.com/api
    analyzer_timeout   CAS_ANALYZER_TIMEOUT   10.0
    gpu_name           KRAUNCHER_GPU_NAME     ""
    gpu_arch           KRAUNCHER_GPU_ARCH     ""
    (task vram_gb)     KRAUNCHER_VRAM_GB      "" (overrides @task(vram_gb=...))
    estimate_only      CAS_ESTIMATE_ONLY      false
    max_task_retries   CAS_MAX_TASK_RETRIES   3
    max_task_chain_sec CAS_MAX_TASK_CHAIN_SEC 0 (= 2x the task's timeout)
    ================== ====================== ========================================

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
        analyzer_url: Any = _UNSET,
        analyzer_timeout: float | None = None,
        gpu_name: str | None = None,
        gpu_arch: str | None = None,
        estimate_only: bool | None = None,
        stream_stderr: bool | None = None,
        send_credentials: bool | None = None,
        max_task_retries: int | None = None,
        max_task_chain_sec: float | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("CAS_API_KEY", "")
        if not self.api_key:
            raise KrauncherError(
                "Missing API key. Pass api_key=... to KrauncherClient(), or set "
                "the CAS_API_KEY environment variable (e.g. in a .env file). "
                "Generate a key at https://krauncher.com → Account → API Keys."
            )
        self.broker_url = (broker_url or os.environ.get("CAS_BROKER_URL", "https://krauncher.com/api")).rstrip("/")

        # Task E2E encryption is mandatory — the broker rejects plaintext
        # submissions. There is no opt-out. The /estimate analyzer call is
        # also always E2E-encrypted; there is no plaintext fallback.

        # analyzer_url is resolved exclusively from the broker (/v1/me).
        # The constructor parameter is kept only for tests / edge cases.
        self._analyzer_url_override = analyzer_url if analyzer_url is not _UNSET else None

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

        # Storage credentials are read from this process's environment and sent
        # to the worker inside the E2E payload. Off means a task with private
        # data fails on download rather than borrowing an unrelated profile.
        if send_credentials is not None:
            self.send_credentials = send_credentials
        else:
            self.send_credentials = os.environ.get(
                "CAS_SEND_CREDENTIALS", "true",
            ).lower() not in ("0", "false", "no")

        # How many times wait() transparently resubmits a task whose failure
        # was our infrastructure's fault (broker-flagged retriable, or a
        # deadline hit before the task ever ran). Not a wall-clock budget:
        # each attempt is a fresh task_id.
        if max_task_retries is not None:
            self.max_task_retries = max_task_retries
        else:
            self.max_task_retries = int(os.environ.get("CAS_MAX_TASK_RETRIES", "3"))

        # Wall-clock ceiling on a whole chain of attempts, in seconds. The
        # attempt counter alone does not bound how long the caller waits: with
        # max_task_retries=3 a chain could run 4x the task's own timeout.
        # 0 means "derive from the timeout" — see _chain_budget().
        if max_task_chain_sec is not None:
            self.max_task_chain_sec = float(max_task_chain_sec)
        else:
            self.max_task_chain_sec = float(
                os.environ.get("CAS_MAX_TASK_CHAIN_SEC", "0")
            )

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
        config = self._get_broker_config()
        token = config.get("analyzer_token")
        # Analyzer choice: absent on an older broker means "off" — never assume
        # consent to send code to an external analyzer.
        llm_backend = config.get("llm_analyzer") or "off"
        user_id = config.get("user_id")
        # Re-create client if URL changed
        if self._analyzer_client is not None and self._analyzer_client._url == url.rstrip("/"):
            self._analyzer_client._llm_backend = llm_backend
            self._analyzer_client._user_id = user_id
            return self._analyzer_client
        self._analyzer_client = AnalyzerClient(
            analyzer_url=url,
            timeout=self._analyzer_timeout,
            token=token,
            user_id=user_id,
            llm_backend=llm_backend,
        )
        return self._analyzer_client

    async def _resolve_dataset_mb(
        self,
        data: str | None,
    ) -> float | None:
        """Query broker for input dataset size (MB) for CU estimation.

        Only includes input data sources — output sources are
        excluded because they don't affect training iteration count.
        Returns None if no data source is specified or on any error (best-effort).
        """
        return await self._resolve_sizes([n for n in (data,) if n is not None])

    async def _resolve_sizes(self, names: list[str]) -> float | None:
        """Total size (MB) of registered data sources, best-effort."""
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
        group_id: str | None = None,
        provider: str | None = None,
        disk_gb: int = 10,
        dataset_size: float | None = None,
        stream_stderr: bool | None = None,
        artifacts: bool = False,
    ) -> Callable:
        """Decorator that marks a function as a remote GPU task.

        KRAUNCHER_VRAM_GB (env) overrides the declared ``vram_gb`` — lets a
        measurement campaign re-target the same calibration samples to a
        different VRAM class without editing the sample files.

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
            artifacts: Return the files the task writes beside itself, in its
                working directory, which is also its ``HOME``
                (``result.files`` / ``result.download()``).  Hidden files and
                directories are skipped — they are caches libraries drop in
                ``~``, not task output.  Artifacts share the result's inline
                size budget; for anything large use a data source instead.
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
                if "files" in sig.parameters:
                    raise KrauncherError(
                        f"task {func.__name__!r} declares a parameter named "
                        f"'files', which collides with the call-time channel "
                        f"for sending files to the task. Rename the parameter."
                    )
            except (ValueError, TypeError):
                pass

            # Cache analyzer result per decorated function — the code_string
            # never changes, so re-analyzing on every call is wasteful and
            # causes timeouts under concurrent load.
            _cached_classification: list[TaskClassification | None] = [None]

            @functools.wraps(func)
            async def wrapper(**kwargs: Any) -> TaskHandle:
                # files= travels beside the code, not as a task argument.
                files = kwargs.pop("files", None)
                return await client._submit(
                    code_string, entry_point, kwargs,
                    func_defaults=_func_defaults,
                    classification_cache=_cached_classification,
                    vram_gb=vram_gb, gpu_arch=gpu_arch, gpu_name=gpu_name,
                    pip=pip, timeout=timeout, priority=priority,
                    data_urls=data_urls, data=data, output=output,
                    group_id=group_id, provider=provider,
                    disk_gb=disk_gb, dataset_size=dataset_size,
                    stream_stderr=stream_stderr, artifacts=artifacts,
                    files=files,
                )

            # Store metadata for introspection and group envelopes
            wrapper._krauncher_code = code_string
            wrapper._krauncher_entry_point = entry_point
            wrapper._krauncher_pip = pip or []
            wrapper._krauncher_provider = provider
            wrapper._krauncher_defaults = _func_defaults
            wrapper._krauncher_cls_cache = _cached_classification
            # Full decorator options, keyed exactly as _submit() parameters —
            # group.submit() forwards them verbatim.
            wrapper._krauncher_options = {
                "vram_gb": vram_gb, "gpu_arch": gpu_arch, "gpu_name": gpu_name,
                "pip": pip, "timeout": timeout, "priority": priority,
                "data_urls": data_urls, "data": data, "output": output,
                "provider": provider, "disk_gb": disk_gb,
                "dataset_size": dataset_size, "stream_stderr": stream_stderr,
                "artifacts": artifacts,
            }

            return wrapper

        return decorator

    async def group(self, *tasks: Callable, name: str | None = None) -> TaskGroup:
        """Build a :class:`TaskGroup` — a shared-requirements envelope for
        tasks that should share one warm worker (Tier-1 group affinity).

        Classifies each member's code (analysis phase only, nothing is
        submitted) and derives what the group's worker must satisfy:

        - VRAM floor = max over members (explicit ``vram_gb`` pins get the
          usual 10% headroom, unpinned members are classified);
        - ``gpu_name`` / ``gpu_arch`` / ``provider`` — shared; conflicting
          explicit pins raise immediately;
        - disk envelope = max member ``disk_gb`` + total size of all members'
          data sources, so the group's data fits the host.

        Submit members with ``await group.submit(task, **kwargs)`` or pass
        ``group=group`` to :meth:`run_code`.
        """
        import math
        import uuid as _uuid

        if not tasks:
            raise KrauncherError("client.group() needs at least one @client.task function")
        vram_floor = 0
        pins: dict[str, set] = {"gpu_name": set(), "gpu_arch": set(), "provider": set()}
        data_names: set[str] = set()
        disk_gb = 0
        for t in tasks:
            opts = getattr(t, "_krauncher_options", None)
            if opts is None:
                raise KrauncherError(
                    "client.group() expects @client.task-decorated functions"
                )
            if opts["vram_gb"] is not None:
                vram = classify_explicit(opts["vram_gb"]).min_vram_gb
            else:
                cls = await self._classify(t._krauncher_code, t._krauncher_defaults or {})
                vram = cls.min_vram_gb
            vram_floor = max(vram_floor, vram)
            for key, bag in pins.items():
                if opts.get(key):
                    bag.add(opts[key])
            for key in ("data",):
                if opts.get(key):
                    data_names.add(opts[key])
            disk_gb = max(disk_gb, opts.get("disk_gb") or 0)
        for key, bag in pins.items():
            if len(bag) > 1:
                raise KrauncherError(
                    f"group members pin different {key}: {sorted(bag)} — "
                    f"a group shares one worker"
                )
        total_mb = await self._resolve_sizes(sorted(data_names))
        if total_mb:
            disk_gb += math.ceil(total_mb / 1024)
        return TaskGroup(
            group_id=name or f"kr-{_uuid.uuid4().hex[:8]}",
            client=self,
            vram_floor=vram_floor,
            gpu_name=next(iter(pins["gpu_name"]), None),
            gpu_arch=next(iter(pins["gpu_arch"]), None),
            provider=next(iter(pins["provider"]), None),
            disk_gb=disk_gb or 10,
        )

    async def _submit(
        self,
        code_string: str,
        entry_point: str,
        kwargs: dict[str, Any],
        *,
        func_defaults: dict[str, Any] | None = None,
        classification_cache: list | None = None,
        classification: TaskClassification | None = None,
        group: TaskGroup | None = None,
        vram_gb: int | None = None,
        gpu_arch: str | None = None,
        gpu_name: str | None = None,
        pip: list[str] | None = None,
        timeout: int = 600,
        priority: int = 1,
        data_urls: list[str] | None = None,
        data: str | None = None,
        output: str | None = None,
        group_id: str | None = None,
        provider: str | None = None,
        disk_gb: int = 10,
        dataset_size: float | None = None,
        stream_stderr: bool | None = None,
        artifacts: bool = False,
        files: dict[str, bytes] | None = None,
    ) -> TaskHandle:
        """Submission core shared by :meth:`task` and :meth:`run_code`.

        Two phases: :meth:`_classify` (analysis request; skipped when a
        precomputed *classification* is passed) -> :meth:`_execute`
        (execution request). The ``estimate_only`` guard sits between them.
        """
        import time as _time
        _submit_start = _time.monotonic()
        # Merge defaults with passed kwargs (passed values take priority)
        merged_kwargs = {**(func_defaults or {}), **kwargs}

        if classification is None:
            classification = await self._classify(
                code_string, merged_kwargs,
                vram_gb=vram_gb, data=data,
                dataset_size=dataset_size,
                classification_cache=classification_cache,
            )

        if group is not None:
            # Group envelope: the shared worker was sized for the whole group
            # — raise this task to the group's floor, inherit unset pins.
            if group.vram_floor and classification.min_vram_gb < group.vram_floor:
                import dataclasses
                from .analyzer import _vram_to_tier
                classification = dataclasses.replace(classification)
                classification.min_vram_gb = group.vram_floor
                classification.tier = _vram_to_tier(group.vram_floor)
            if group_id is None:
                group_id = group.group_id
            if gpu_name is None and group.gpu_name:
                gpu_name = group.gpu_name
            if gpu_arch is None and group.gpu_arch:
                gpu_arch = group.gpu_arch
            if provider is None and group.provider:
                provider = group.provider
            disk_gb = max(disk_gb, group.disk_gb)

        if self.estimate_only:
            c = classification
            _logger.info(
                "estimate_only=true — skipping broker submission "
                "(CU=%s, VRAM=%sGB, tier=%s, method=%s, cpu_only=%s)",
                c.compute_units, c.min_vram_gb, c.tier, c.analysis_method, c.cpu_only,
            )
            # Do NOT exit: return a stub handle so the script continues
            # and every decorated function gets its own analyze request
            # (multi-task scripts, e.g. 17 phase1/phase2).
            return _EstimateOnlyHandle(classification)

        return await self._execute(
            code_string, entry_point, kwargs, merged_kwargs, classification,
            gpu_arch=gpu_arch, gpu_name=gpu_name, pip=pip, timeout=timeout,
            priority=priority, data_urls=data_urls, data=data, output=output,
            group_id=group_id, provider=provider,
            disk_gb=disk_gb, stream_stderr=stream_stderr,
            artifacts=artifacts, files=files, submit_start=_submit_start,
        )

    async def _classify(
        self,
        code_string: str,
        merged_kwargs: dict[str, Any],
        *,
        vram_gb: int | None = None,
        data: str | None = None,
        dataset_size: float | None = None,
        classification_cache: list | None = None,
    ) -> TaskClassification:
        """Analysis phase: classify the code via cas-analyzer.

        Resolves the dataset size, applies the explicit/env vram_gb override,
        caches per decorated function, and logs the result at DEBUG level.
        No broker submission happens here.
        """
        client = self
        # Env override for the declared VRAM class (see task() docstring).
        _vram_env = os.environ.get("KRAUNCHER_VRAM_GB", "")
        if _vram_env:
            vram_gb = int(_vram_env)

        # Classification: call analyzer once, cache for subsequent calls.
        if classification_cache is not None and classification_cache[0] is not None:
            classification = classification_cache[0]
        else:
            # _analyzer raises KrauncherError if no analyzer configured
            try:
                # Query broker for data source sizes to improve CU estimation
                dataset_mb = dataset_size or await client._resolve_dataset_mb(data)
                classification = await client._analyzer.classify(
                    code_string, dataset_mb=dataset_mb, kwargs=merged_kwargs,
                )
            except KrauncherError:
                raise
            except Exception as exc:
                raise KrauncherError(
                    f"Analyzer failed and CU estimation is unavailable: {exc}"
                ) from exc
            if classification_cache is not None:
                classification_cache[0] = classification

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
            # Generic pass-through: any unmapped analyzer debug field
            # (cu_prefill/cu_decode and future) prints itself — no per-field code.
            for _k, _v in c.extra_debug.items():
                parts.append(f"{_k}={_v}")
            if c.analyzer_time is not None:
                parts.append(f"time={c.analyzer_time:.2f}s")
            _logger.debug("Classification: %s", ", ".join(parts))

        return classification

    async def _execute(
        self,
        code_string: str,
        entry_point: str,
        kwargs: dict[str, Any],
        merged_kwargs: dict[str, Any],
        classification: TaskClassification,
        *,
        gpu_arch: str | None = None,
        gpu_name: str | None = None,
        pip: list[str] | None = None,
        timeout: int = 600,
        priority: int = 1,
        data_urls: list[str] | None = None,
        data: str | None = None,
        output: str | None = None,
        group_id: str | None = None,
        provider: str | None = None,
        disk_gb: int = 10,
        stream_stderr: bool | None = None,
        artifacts: bool = False,
        files: dict[str, bytes] | None = None,
        submit_start: float | None = None,
    ) -> TaskHandle:
        """Execution phase: build the payload and POST /tasks.

        Takes a ready :class:`TaskClassification` — no analyzer calls here.
        """
        client = self
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

        # E2E encryption (mandatory): generate an ephemeral keypair and withhold
        # the plaintext code+args from the broker — they are uploaded encrypted
        # to the worker via the relay.
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

        # Neither the declaration nor the files are sent here — they are data
        # plane and ride encrypted to the worker with the code (see
        # _relay_stream_sync). The broker sees none of it.
        if files:
            from .values import INLINE_BUDGET_BYTES
            # The code shares the payload with the files, so it counts too —
            # otherwise this passes and the transport fails instead.
            total = sum(len(b) for b in files.values()) + len(code_string.encode())
            if total > INLINE_BUDGET_BYTES:
                raise ValueTransferError(
                    f"code plus files= is {total / (1024 * 1024):.1f} MB — "
                    f"exceeds the {INLINE_BUDGET_BYTES // (1024 * 1024)} MB "
                    f"payload budget. Put data this size in a "
                    f"data source."
                )

        body["classification"] = classification.to_dict()

        async def _post_task(parent_task_id: str | None = None, attempt: int = 1) -> str:
            # A retry says which task it replaces, so the chain of attempts is
            # one story in the task history and the billing rows rather than
            # N unrelated tasks.
            if parent_task_id:
                body["parent_task_id"] = parent_task_id
                body["attempt"] = attempt
            async with httpx.AsyncClient(timeout=30.0) as session:
                resp = await session.post(
                    f"{client.broker_url}/tasks",
                    json=body,
                    headers={"X-API-Key": client.api_key},
                )
                _check_response(resp)
                return resp.json()["task_id"]

        credentials = collect_credentials() if client.send_credentials else {}

        task_id = await _post_task()
        handle = TaskHandle(
            task_id=task_id,
            client=client,
            ek_priv=ek_priv,
            plaintext_code=code_string,
            plaintext_args=kwargs,
            credentials=credentials,
            classification=classification,
            submit_start=submit_start,
            resubmit=_post_task,
            stream_stderr=effective_stream_stderr,
            artifacts=artifacts,
            files=files,
        )
        _inflight.register(handle)
        return handle


    async def run_code(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
        lenient_outputs: bool = False,
        **task_options: Any,
    ) -> "TaskHandle":
        """Run a code block (a notebook cell, an editor selection) remotely.

        The application-agnostic entry point for adapters: *code* becomes the
        body of a generated task function, *inputs* are named values injected
        into its namespace (and visible to the analyzer's CU estimation), and
        the names in *outputs* are collected from the block's namespace and
        returned as the task's output dict — decode it with
        :func:`krauncher.values.decode_outputs`.

        Values must be JSON-safe and fit the inline budget together with the
        code (see ``krauncher.values``); larger data goes through a
        data source.

        Args:
            code: The code block to execute remotely.
            inputs: ``{name: value}`` injected as the block's variables.
            outputs: Variable names to return from the block's namespace.
            lenient_outputs: When ``True`` (auto-detected outputs), names
                that are unset or non-JSON-safe are dropped remotely instead
                of failing the task.
            **task_options: Same options as :meth:`task` (``pip``, ``timeout``,
                ``vram_gb``, ``gpu_name``, ...), plus
                ``classification=`` — a precomputed :class:`TaskClassification`
                from :meth:`estimate_code` (skips the analysis phase) — and
                ``group=`` — a :class:`TaskGroup` from :meth:`group` for
                warm-worker co-location.

        Returns:
            A :class:`TaskHandle`; ``result.output`` is the outputs dict.
        """
        source, entry_point, kwargs = self._prepare_code_block(
            code, inputs, outputs, lenient_outputs=lenient_outputs,
        )
        return await self._submit(source, entry_point, kwargs, **task_options)

    async def estimate_code(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
        lenient_outputs: bool = False,
        vram_gb: int | None = None,
        data: str | None = None,
        dataset_size: float | None = None,
    ) -> TaskClassification:
        """Analysis request for a code block — classify without submitting.

        Synthesizes exactly the source :meth:`run_code` would submit, calls
        the analyzer, and returns the :class:`TaskClassification`. Pass the
        result to ``run_code(..., classification=...)`` to execute without
        a second analysis.
        """
        source, _entry, kwargs = self._prepare_code_block(
            code, inputs, outputs, lenient_outputs=lenient_outputs,
        )
        return await self._classify(
            source, kwargs,
            vram_gb=vram_gb, data=data,
            dataset_size=dataset_size,
        )

    def _prepare_code_block(
        self,
        code: str,
        inputs: dict[str, Any] | None,
        outputs: list[str] | None,
        *,
        lenient_outputs: bool = False,
    ) -> tuple[str, str, dict[str, Any]]:
        """Shared front half of :meth:`run_code` / :meth:`estimate_code`:
        synthesize the task source and encode inputs under the inline budget.
        """
        from .codeblock import build_code_source
        from .values import INLINE_BUDGET_BYTES, encode_inputs

        inputs = inputs or {}
        budget = INLINE_BUDGET_BYTES - len(code.encode("utf-8"))
        if budget <= 0:
            raise ValueTransferError(
                f"code block alone exceeds the "
                f"{INLINE_BUDGET_BYTES / (1024 * 1024):.1f} MB inline budget"
            )
        kwargs = encode_inputs(list(inputs), inputs, limit_bytes=budget)
        source, entry_point = build_code_source(
            code, list(inputs), outputs or [], lenient_outputs=lenient_outputs,
        )
        return source, entry_point, kwargs

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
