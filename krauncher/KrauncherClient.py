"""KrauncherClient — main entry point for submitting GPU tasks."""

from __future__ import annotations

import functools
from typing import Any, Callable

import httpx

from .exceptions import KrauncherError
from .models import Runner, TaskHandle, _check_response
from .serializer import serialize_function


class KrauncherClient:
    """Client for submitting tasks to the CaS broker.

    Usage::

        client = KrauncherClient(api_key="cas_...", broker_url="http://...")

        @client.task(vram_gb=24, timeout=3600)
        def train(data):
            import torch
            return {"loss": 0.01}

        handle = await train(data={"epochs": 5})
        result = await handle
    """

    def __init__(self, api_key: str, broker_url: str = "http://localhost:8000") -> None:
        self.api_key = api_key
        self.broker_url = broker_url.rstrip("/")

    def task(
        self,
        *,
        vram_gb: int = 8,
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
            vram_gb: Minimum GPU VRAM in GB.
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

            @functools.wraps(func)
            async def wrapper(**kwargs: Any) -> TaskHandle:
                requirements: dict[str, Any] = {
                    "min_vram_gb": vram_gb,
                    "gpu_arch": gpu_arch,
                }
                if provider is not None:
                    requirements["provider_name"] = provider

                body: dict[str, Any] = {
                    "priority": priority,
                    "requirements": requirements,
                    "payload": {
                        "code_string": code_string,
                        "entry_point": entry_point,
                        "args": kwargs,
                        "pip": pip or [],
                    },
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

                async with httpx.AsyncClient(timeout=30.0) as session:
                    resp = await session.post(
                        f"{client.broker_url}/tasks",
                        json=body,
                        headers={"X-API-Key": client.api_key},
                    )
                    _check_response(resp)
                    task_id = resp.json()["task_id"]
                    return TaskHandle(task_id=task_id, client=client)

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
