"""TaskHandle, TaskResult and Runner models."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import httpx

from .exceptions import AuthError, KrauncherError, RemoteTimeout, TaskError, TaskTimeout

if TYPE_CHECKING:
    from .KrauncherClient import KrauncherClient

TERMINAL_STATUSES = frozenset({"completed", "failed", "timeout", "hardware_preempted"})

_STATUS_SYMBOL = {
    "available": "✓",
    "busy": "●",
    "provisioning": "◌",
    "draining": "↓",
    "offline": "✗",
}


@dataclass(frozen=True)
class Runner:
    """A compute host available in the fleet.

    Returned by :meth:`KrauncherClient.list_runners`.
    Pass ``runner.provider`` to ``@client.task(provider=...)`` to pin a task
    to this specific provider.
    """

    provider: str
    host_id: str
    gpu_model: str
    gpu_count: int
    vram_gb: int
    gpu_arch: str
    price_per_hour_usd: float
    status: str
    spot: bool
    region: str
    worker_id: str | None = None

    def __str__(self) -> str:
        spot_tag = " (spot)" if self.spot else ""
        price = f"${self.price_per_hour_usd:.2f}/hr" if self.price_per_hour_usd else "free"
        symbol = _STATUS_SYMBOL.get(self.status, "?")
        return (
            f"{symbol} [{self.provider}] {self.gpu_model} "
            f"{self.vram_gb}GB {self.gpu_arch} — {price}{spot_tag} — {self.status}"
        )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskResult:
    """Parsed result of a completed task."""

    task_id: str
    status: str
    worker_id: str = ""
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    traceback: str | None = None
    exit_code: int = 0
    actual_gpu: str = "unknown"
    execution_time_sec: float = 0.0
    duration_sec: float = 0.0
    gpu_util_avg: float = 0.0
    cost_usd: float = 0.0
    queue_wait_sec: float = 0.0
    download_sec: float = 0.0
    pip_install_sec: float = 0.0

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> TaskResult:
        """Parse from GET /tasks/{id} response with result present."""
        result = data.get("result") or {}
        sys_info = result.get("system_info") or {}
        exec_result = result.get("execution_result") or {}
        billing = result.get("billing_metrics") or {}

        duration = billing.get("duration_sec", 0.0)
        price = billing.get("price_per_hour_usd", 0.0)

        return cls(
            task_id=str(data["task_id"]),
            status=data["status"],
            worker_id=data.get("worker_id", ""),
            output=exec_result.get("output"),
            stdout=exec_result.get("stdout", ""),
            stderr=exec_result.get("stderr", ""),
            traceback=exec_result.get("traceback"),
            exit_code=sys_info.get("exit_code", 0),
            actual_gpu=sys_info.get("actual_gpu", "unknown"),
            execution_time_sec=sys_info.get("execution_time_sec", 0.0),
            duration_sec=duration,
            gpu_util_avg=billing.get("gpu_util_avg", 0.0),
            cost_usd=duration * price / 3600.0 if duration and price else 0.0,
            queue_wait_sec=data.get("queue_wait_sec") or 0.0,
            download_sec=data.get("download_sec") or 0.0,
            pip_install_sec=data.get("pip_install_sec") or 0.0,
        )


def _check_response(resp: httpx.Response) -> None:
    """Raise appropriate exception for error HTTP responses."""
    if resp.status_code in (401, 403):
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise AuthError(f"Authentication failed ({resp.status_code}): {detail}")

    if resp.status_code == 404:
        raise KrauncherError("Task not found (404)")

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise KrauncherError(f"Broker returned {resp.status_code}: {detail}")


async def _relay_stream(
    task_id: str,
    relay_url: str,
    token: str,
    on_log: Callable[[dict[str, Any]], None],
) -> None:
    """Connect to relay WebSocket and feed messages to *on_log* until stream ends.

    Silently exits on any connection error — the main wait() loop continues
    polling normally, providing automatic fallback when relay is unavailable.
    """
    try:
        import websockets
        import websockets.exceptions
    except ImportError:
        logger.debug("relay_streaming_unavailable: websockets not installed")
        return

    ws_url = f"{relay_url.rstrip('/')}/tasks/{task_id}/stream?token={token}"
    try:
        async with websockets.connect(ws_url) as ws:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                try:
                    on_log(msg)
                except Exception:
                    pass
                # Relay closes after stream_ended grace period;
                # we also exit early on the event to be responsive.
                if (
                    msg.get("type") == "event"
                    and isinstance(msg.get("data"), dict)
                    and msg["data"].get("name") == "stream_ended"
                ):
                    break
    except Exception as exc:
        logger.debug("relay_stream_error: %s", exc)


class TaskHandle:
    """Async handle to a submitted task.

    Usage::

        task = await my_func(x=42)    # submit, get handle
        result = await task            # async wait for completion
    """

    def __init__(self, task_id: str, client: KrauncherClient) -> None:
        self.task_id = task_id
        self._client = client
        self._result: TaskResult | None = None

    def __repr__(self) -> str:
        return f"TaskHandle(task_id={self.task_id!r})"

    def __await__(self):
        """Allow ``result = await task``."""
        return self.wait().__await__()

    async def wait(
        self,
        *,
        timeout: float = 600.0,
        on_log: Callable[[dict[str, Any]], None] | None = None,
    ) -> TaskResult:
        """Async poll until terminal status. Adaptive delay 0.5s -> 5s.

        If the broker returns a ``relay_url`` + ``relay_task_token`` for an
        active task *and* ``on_log`` is provided, a concurrent WebSocket
        subscription to the relay is opened.  Each relay message (stdout,
        stderr, event, metric) is passed to ``on_log`` in real time.

        ``on_log`` signature::

            def on_log(msg: dict) -> None:
                # msg keys: task_id, type, ts, seq, data
                if msg["type"] in ("stdout", "stderr"):
                    print(msg["data"].get("text", ""), end="")

        Args:
            timeout: Client-side wall-clock timeout in seconds.
            on_log: Optional callback for real-time log messages from relay.

        Returns:
            TaskResult for completed tasks.

        Raises:
            TaskError: If the task failed or was preempted.
            TaskTimeout: If timeout exceeded.
        """
        if self._result is not None:
            return self._check_result(self._result)

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        delay = 0.5
        relay_task: asyncio.Task | None = None

        async with httpx.AsyncClient(timeout=30.0) as session:
            while True:
                data = await self._poll(session)

                # Start relay streaming on first poll that returns relay info
                if (
                    relay_task is None
                    and on_log is not None
                    and data.get("relay_url")
                    and data.get("relay_task_token")
                    and data["status"] not in TERMINAL_STATUSES
                ):
                    relay_task = asyncio.create_task(
                        _relay_stream(
                            task_id=self.task_id,
                            relay_url=data["relay_url"],
                            token=data["relay_task_token"],
                            on_log=on_log,
                        ),
                        name=f"relay-{self.task_id[:8]}",
                    )

                if data["status"] in TERMINAL_STATUSES:
                    if relay_task is not None:
                        # Wait briefly for relay to drain final messages
                        try:
                            await asyncio.wait_for(asyncio.shield(relay_task), timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                            relay_task.cancel()

                    if data.get("result") is not None:
                        self._result = TaskResult.from_response(data)
                    else:
                        # Terminal status but result not stored yet — retry briefly
                        for _ in range(3):
                            await asyncio.sleep(0.5)
                            data = await self._poll(session)
                            if data.get("result") is not None:
                                self._result = TaskResult.from_response(data)
                                break
                        else:
                            self._result = TaskResult(
                                task_id=self.task_id, status=data["status"],
                            )

                    return self._check_result(self._result)

                remaining = deadline - loop.time()
                if remaining <= 0:
                    if relay_task is not None:
                        relay_task.cancel()
                    raise TaskTimeout(self.task_id, timeout)

                await asyncio.sleep(min(delay, remaining))
                delay = min(delay * 1.5, 5.0)


    async def status(self) -> dict[str, Any]:
        """Single poll — return raw status dict from broker."""
        async with httpx.AsyncClient(timeout=30.0) as session:
            return await self._poll(session)

    def done(self) -> bool:
        """Non-blocking check using cached result."""
        return self._result is not None

    @property
    def result(self) -> TaskResult | None:
        """Cached result, or None if not yet completed."""
        return self._result

    async def _poll(self, session: httpx.AsyncClient) -> dict[str, Any]:
        """GET /tasks/{task_id} with auth headers."""
        resp = await session.get(
            f"{self._client.broker_url}/tasks/{self.task_id}",
            headers={"X-API-Key": self._client.api_key},
        )
        _check_response(resp)
        return resp.json()

    @staticmethod
    def _check_result(result: TaskResult) -> TaskResult:
        """Raise TaskError for non-completed results, return otherwise."""
        if result.status == "completed":
            return result

        if result.status == "timeout":
            raise RemoteTimeout(task_id=result.task_id)

        if result.status == "failed":
            raise TaskError(
                f"Task {result.task_id} failed",
                task_id=result.task_id,
                remote_traceback=result.traceback or result.stderr or None,
            )

        raise TaskError(
            f"Task {result.task_id} was preempted ({result.status})",
            task_id=result.task_id,
            remote_traceback=None,
        )
