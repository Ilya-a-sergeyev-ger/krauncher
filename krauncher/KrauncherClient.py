"""KrauncherClient — main entry point for submitting GPU tasks."""

from __future__ import annotations

import functools
from typing import Any, Callable

import httpx

from .exceptions import KrauncherError
from .models import TaskHandle, _check_response
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
        """

        client = self

        def decorator(func: Callable) -> Callable:
            # Serialize at decoration time — fail fast on invalid functions
            code_string, entry_point = serialize_function(func)

            @functools.wraps(func)
            async def wrapper(**kwargs: Any) -> TaskHandle:
                body = {
                    "priority": priority,
                    "requirements": {
                        "min_vram_gb": vram_gb,
                        "gpu_arch": gpu_arch,
                    },
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

            return wrapper

        return decorator
