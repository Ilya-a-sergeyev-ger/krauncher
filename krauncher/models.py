# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""TaskHandle, TaskResult and Runner models."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import httpx

from .exceptions import AuthError, E2EIdentityMismatch, InsufficientBalanceError, KrauncherError, NoCapacityError, PayloadDeliveryError, RemoteTimeout, TaskError, TaskTimeout

if TYPE_CHECKING:
    from .KrauncherClient import KrauncherClient

TERMINAL_STATUSES = frozenset({"completed", "failed", "timeout", "hardware_preempted", "no_capacity"})

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

# Dedicated thread pool for blocking gRPC relay streams.
# The default executor is min(32, cpu_count+4) which saturates when many
# tasks run concurrently — queued threads can't receive key_exchange in
# time, causing e2e_payload_timeout on the worker side.
_relay_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=128,
    thread_name_prefix="relay-stream",
)


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
    provider_cost: float = 0.0      # raw provider cost in client currency (no markup, no fee)
    actual_cu: float = 0.0          # actual compute units (from real execution time + GPU)
    charged_ku: float = 0.0         # KU charged for compute (with markup, no fee)
    charged_local: float = 0.0      # compute charge in client currency (with markup, no fee)
    fee_ku: float = 0.0             # dispatch fee in KU
    fee_local: float = 0.0          # dispatch fee in client currency
    total_charged_ku: float = 0.0   # charged_ku + fee_ku (full deduction from balance)
    total_charged_local: float = 0.0  # charged_local + fee_local
    billing_currency: str = "USD"
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
            provider_cost=data.get("provider_cost_local") or data.get("cost_usd") or 0.0,
            actual_cu=data.get("actual_cu") or 0.0,
            charged_ku=data.get("charged_ku") or 0.0,
            charged_local=data.get("client_cost_local") or 0.0,
            fee_ku=data.get("fee_ku") or 0.0,
            fee_local=data.get("fee_local") or 0.0,
            total_charged_ku=data.get("total_charged_ku") or 0.0,
            total_charged_local=data.get("total_charged_local") or 0.0,
            billing_currency=data.get("billing_currency") or "USD",
            queue_wait_sec=data.get("queue_wait_sec") or 0.0,
            download_sec=data.get("download_sec") or 0.0,
            pip_install_sec=data.get("pip_install_sec") or 0.0,
        )


_CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£"}


def _fmt_money(amount: float, currency: str) -> str:
    """Format a money amount for the user-facing summary."""
    sym = _CURRENCY_SYMBOLS.get(currency, "")
    if sym:
        return f"{sym}{amount:.4f}"
    return f"{amount:.4f} {currency}"


def _log_billing_summary(result: TaskResult) -> None:
    """Print a concise billing breakdown after a task completes.

    Format::

        Task <short_id> done in 12.3s — compute 25.0 KU (€0.02), \
        fee 1.2 KU (€0.001), total 26.2 KU (€0.021)

    Skipped when there is nothing to charge (no_capacity, dead_letter w/o
    fee, legacy tasks without billing fields).
    """
    if result.total_charged_ku <= 0 and result.charged_ku <= 0 and result.fee_ku <= 0:
        return

    cur = result.billing_currency or "USD"
    duration = result.duration_sec or result.execution_time_sec or 0.0

    parts = [f"Task {result.task_id[:8]} done in {duration:.1f}s"]

    compute_ku = result.charged_ku
    compute_local = result.charged_local or result.provider_cost
    if compute_ku > 0 or compute_local > 0:
        parts.append(f"compute {compute_ku:.4f} KU ({_fmt_money(compute_local, cur)})")

    if result.fee_ku > 0:
        parts.append(f"fee {result.fee_ku:.4f} KU ({_fmt_money(result.fee_local, cur)})")

    total_ku = result.total_charged_ku or (compute_ku + result.fee_ku)
    total_local = result.total_charged_local or (compute_local + result.fee_local)
    parts.append(f"total {total_ku:.4f} KU ({_fmt_money(total_local, cur)})")

    logger.info(" — ".join(parts))


def _check_response(resp: httpx.Response) -> None:
    """Raise appropriate exception for error HTTP responses."""
    if resp.status_code in (401, 403):
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise AuthError(f"Authentication failed ({resp.status_code}): {detail}")

    if resp.status_code == 402:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise InsufficientBalanceError(detail)

    if resp.status_code == 404:
        raise KrauncherError("Task not found (404)")

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise KrauncherError(f"Broker returned {resp.status_code}: {detail}")


# ---------------------------------------------------------------------------
# Synchronous relay stream — runs in a dedicated thread per task.
#
# Why synchronous gRPC instead of grpc.aio?
#
# grpc.aio uses an internal PollerCompletionQueue that registers a
# single fd (pipe/eventfd) with the asyncio event loop.  When multiple
# threads each create their own asyncio event loop (as tutorial 08 does
# with ThreadPoolExecutor + asyncio.new_event_loop()), multiple pollers
# race for the same internal fd, causing:
#
#   BlockingIOError: [Errno 11] Resource temporarily unavailable
#
# at grpc._cython.cygrpc.PollerCompletionQueue._handle_events.
# Messages are silently lost — TaskStream never delivers key_exchange
# to most subscribers, and the worker times out waiting for payload.
#
# Solution: use plain synchronous grpc.insecure_channel + blocking
# iteration (for msg in stub.TaskStream(...)).  Each task gets its own
# thread via loop.run_in_executor().  All E2E operations (key exchange,
# derive, encrypt, UploadPayload) happen on the same channel in the
# same thread — simple, atomic, no cross-thread conflicts.
# ---------------------------------------------------------------------------

_TLS_SCHEMES = ("grpcs://", "https://", "wss://")
_PLAINTEXT_SCHEMES = ("grpc://", "ws://", "http://")


def _detect_relay_tls(raw_url: str) -> bool:
    """Decide TLS vs plaintext for the relay channel.

    TLS is enabled if **either** the URL scheme implies it (grpcs://, https://,
    wss://) **or** KRAUNCHER_RELAY_TLS env is set to a truthy value
    (1/true/yes/on, case-insensitive).
    """
    import os
    if raw_url.startswith(_TLS_SCHEMES):
        return True
    val = (os.environ.get("KRAUNCHER_RELAY_TLS") or "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _make_relay_channel(target: str, use_tls: bool, ca_pem: str | None = None):
    """Open a gRPC channel to the relay. *target* must be bare host:port.

    *ca_pem* is the private CA cert delivered by the broker (in-memory). It takes
    precedence over the KRAUNCHER_RELAY_CA file path, so clients need no local
    cert file — the broker is the trust distribution point.
    """
    import os
    import grpc
    if use_tls:
        ca_bytes = ca_pem.encode() if ca_pem else None
        if ca_bytes is None:
            ca_path = os.environ.get("KRAUNCHER_RELAY_CA") or ""
            if ca_path:
                with open(ca_path, "rb") as f:
                    ca_bytes = f.read()
        # Relays are addressed by IP (no DNS); validate the cert against a fixed
        # logical name carried in the cert SAN instead of the target host, so one
        # private CA can cover any number of relays without DNS.
        authority = (os.environ.get("KRAUNCHER_RELAY_AUTHORITY") or "cas-relay").strip()
        options = [("grpc.ssl_target_name_override", authority)] if authority else None
        creds = grpc.ssl_channel_credentials(root_certificates=ca_bytes)
        return grpc.secure_channel(target, creds, options=options)
    return grpc.insecure_channel(target)


def _relay_stream_sync(
    task_id: str,
    target: str,
    token: str,
    on_log: Callable[[dict[str, Any]], None],
    *,
    ek_priv: Any = None,
    plaintext_code: str | None = None,
    plaintext_args: dict[str, Any] | None = None,
    expected_worker_pub: str | None = None,
    use_tls: bool = False,
    ca_pem: str | None = None,
) -> None:
    """Blocking gRPC relay stream — meant to run in a worker thread."""
    try:
        import grpc
        from . import relay_pb2, relay_pb2_grpc
    except ImportError as exc:
        logger.debug("[relay] import error: %s", exc)
        return

    e2e_mode = ek_priv is not None
    logger.debug(
        "[relay] connecting task_id=%s target=%s tls=%s e2e=%s",
        task_id[:8], target, use_tls, e2e_mode,
    )

    metadata = [("authorization", f"bearer {token}")]
    shared_key: bytes | None = None

    try:
        with _make_relay_channel(target, use_tls, ca_pem) as channel:
            stub = relay_pb2_grpc.RelayStub(channel)
            logger.debug("[relay] TaskStream open task_id=%s", task_id[:8])

            for proto_msg in stub.TaskStream(
                relay_pb2.TaskStreamRequest(task_id=task_id),
                metadata=metadata,
            ):
                try:
                    data_parsed = json.loads(proto_msg.data) if proto_msg.data else {}
                except (json.JSONDecodeError, TypeError):
                    data_parsed = {}

                msg: dict[str, Any] = {
                    "task_id": proto_msg.task_id,
                    "type": proto_msg.type,
                    "ts": proto_msg.ts,
                    "seq": proto_msg.seq,
                    "data": data_parsed,
                }

                logger.debug(
                    "[relay] msg seq=%d type=%s data_keys=%s",
                    proto_msg.seq,
                    proto_msg.type,
                    list(data_parsed.keys()) if isinstance(data_parsed, dict) else type(data_parsed).__name__,
                )

                # --- E2E: key_exchange handling ---
                if (
                    ek_priv is not None
                    and shared_key is None
                    and msg["type"] == "event"
                    and isinstance(msg["data"], dict)
                    and msg["data"].get("name") == "key_exchange"
                ):
                    logger.debug("[relay] key_exchange received task_id=%s", task_id[:8])
                    try:
                        import base64
                        import hmac
                        import os
                        from .crypto import derive_shared_secret, encrypt
                        wk_pub_b64 = msg["data"]["pub"]

                        # P0-SEC-2 Variant A: identity-bind worker pubkey via
                        # broker-trusted channel. Reject mismatches in strict
                        # mode, warn-only otherwise (rollout flag).
                        if expected_worker_pub:
                            if not hmac.compare_digest(wk_pub_b64, expected_worker_pub):
                                strict = os.environ.get("KRAUNCHER_E2E_STRICT") == "1"
                                logger.warning(
                                    "[relay] E2E pubkey mismatch task_id=%s "
                                    "relay_pub=%s... broker_pub=%s... strict=%s",
                                    task_id[:8],
                                    wk_pub_b64[:12],
                                    expected_worker_pub[:12],
                                    strict,
                                )
                                if strict:
                                    raise E2EIdentityMismatch(task_id)

                        wk_pub_bytes = base64.urlsafe_b64decode(wk_pub_b64 + "==")
                        shared_key = derive_shared_secret(ek_priv, wk_pub_bytes)
                        logger.debug("[relay] shared key derived task_id=%s", task_id[:8])

                        payload_plain = json.dumps({
                            "code_string": plaintext_code or "",
                            "args": plaintext_args or {},
                        }).encode()
                        enc_payload = encrypt(shared_key, payload_plain)
                        logger.debug(
                            "[relay] uploading payload task_id=%s plain_len=%d enc_len=%d",
                            task_id[:8], len(payload_plain), len(enc_payload),
                        )

                        stub.UploadPayload(
                            relay_pb2.UploadPayloadRequest(
                                task_id=task_id,
                                data=json.dumps({"enc": enc_payload}).encode(),
                            ),
                            metadata=metadata,
                        )
                        logger.debug("[relay] payload uploaded ok task_id=%s", task_id[:8])
                    except E2EIdentityMismatch:
                        # Strict-mode security failure — surface to caller.
                        raise
                    except Exception as exc:
                        logger.debug(
                            "[relay] key_exchange error task_id=%s: %s: %s",
                            task_id[:8], type(exc).__name__, exc,
                        )
                    continue

                # --- E2E: decrypt data field for subsequent messages ---
                if shared_key is not None:
                    data_field = msg.get("data")
                    if isinstance(data_field, dict) and "enc" in data_field:
                        try:
                            from .crypto import decrypt
                            plaintext = decrypt(shared_key, data_field["enc"])
                            msg = dict(msg)
                            msg["data"] = json.loads(plaintext)
                        except Exception as exc:
                            logger.debug("[relay] decrypt error: %s", exc)
                            continue

                # --- Deliver to caller ---
                try:
                    on_log(msg)
                except Exception:
                    pass

                # stream_ended is always sent plaintext; exit relay loop promptly
                if (
                    msg.get("type") == "event"
                    and isinstance(msg.get("data"), dict)
                    and msg["data"].get("name") == "stream_ended"
                ):
                    logger.debug("[relay] stream_ended task_id=%s", task_id[:8])
                    break

    except E2EIdentityMismatch:
        raise
    except Exception as exc:
        logger.debug("[relay] error task_id=%s: %s: %s", task_id[:8], type(exc).__name__, exc)


async def _relay_stream(
    task_id: str,
    relay_url: str,
    token: str,
    on_log: Callable[[dict[str, Any]], None],
    *,
    ek_priv: Any = None,
    plaintext_code: str | None = None,
    plaintext_args: dict[str, Any] | None = None,
    expected_worker_pub: str | None = None,
    ca_pem: str | None = None,
) -> None:
    """Async wrapper: runs synchronous gRPC relay in a thread pool executor.

    relay_url format: "host:port"  (scheme prefix is stripped if present).

    Silently exits on any connection error — the main wait() loop continues
    polling normally, providing automatic fallback when relay is unavailable.
    """
    # Decide TLS from raw URL scheme (or env), then strip scheme for grpc target.
    use_tls = _detect_relay_tls(relay_url)
    target = relay_url
    for prefix in _TLS_SCHEMES + _PLAINTEXT_SCHEMES:
        if target.startswith(prefix):
            target = target[len(prefix):]
            break
    target = target.rstrip("/")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _relay_executor,
        lambda: _relay_stream_sync(
            task_id,
            target,
            token,
            on_log,
            ek_priv=ek_priv,
            plaintext_code=plaintext_code,
            plaintext_args=plaintext_args,
            expected_worker_pub=expected_worker_pub,
            use_tls=use_tls,
            ca_pem=ca_pem,
        ),
    )


class TaskHandle:
    """Async handle to a submitted task.

    Usage::

        task = await my_func(x=42)    # submit, get handle
        result = await task            # async wait for completion
    """

    def __init__(
        self,
        task_id: str,
        client: KrauncherClient,
        ek_priv: Any = None,
        plaintext_code: str | None = None,
        plaintext_args: dict[str, Any] | None = None,
        classification: Any = None,
        submit_start: float | None = None,
        resubmit: Callable[[], Any] | None = None,
        stream_stderr: bool = False,
    ) -> None:
        self.task_id = task_id
        self._client = client
        self._result: TaskResult | None = None
        self._last_status: str = ""
        self._phase_waiting_logged: bool = False
        self._phase_host_logged: bool = False
        self._phase_provisioning_logged: bool = False
        self._phase_executing_logged: bool = False
        self._phase_downloading_logged: bool = False
        self._initial_hw: str = ""  # first host info for change detection
        self._host_change_logged: bool = False
        self._pending_download_event: dict | None = None  # buffered download_started
        self._submit_start: float | None = submit_start
        self._waiting_start: float | None = None  # time when "Waiting for worker" was logged
        # E2E fields — set when client.encrypt=True
        self._ek_priv = ek_priv
        self._plaintext_code = plaintext_code
        self._plaintext_args = plaintext_args
        #: Task classification result (TaskClassification or None)
        self.classification = classification
        # Re-submit callback: returns a new task_id when invoked. Used by wait()
        # to transparently retry on no_capacity. If None, NoCapacityError is
        # raised to the caller so they can implement their own policy.
        self._resubmit = resubmit
        self._no_capacity_attempts: int = 0
        #: When true and wait() is called without on_log, install a default handler
        #: that mirrors remote stdout/stderr to local sys.stdout/sys.stderr.
        self.stream_stderr = stream_stderr

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

        In E2E mode (client.encrypt=True), the relay stream is used to perform
        the key exchange and deliver the encrypted task payload to the worker.
        Relay messages are transparently decrypted before being passed to on_log.
        When E2E is active, wait() automatically opens the relay stream even if
        on_log is not provided (needed to deliver the payload).

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

        # When stream_stderr is on and the caller didn't provide on_log, mirror
        # remote stdout/stderr to local sys.stdout/sys.stderr.
        if on_log is None and self.stream_stderr:
            import sys as _sys

            def _console_on_log(msg: dict) -> None:
                t = msg.get("type")
                if t == "stdout":
                    _sys.stdout.write((msg.get("data") or {}).get("text", ""))
                    _sys.stdout.flush()
                elif t == "stderr":
                    _sys.stderr.write((msg.get("data") or {}).get("text", ""))
                    _sys.stderr.flush()

            on_log = _console_on_log

        # In E2E mode we must open relay even without on_log (to deliver payload)
        needs_relay = on_log is not None or self._ek_priv is not None
        _user_on_log = on_log or (lambda _msg: None)

        def _on_log_effective(msg: dict) -> None:
            if msg.get("type") == "event":
                event_data = (msg.get("data") or {})
                name = event_data.get("name", "")
                if name == "worker_ready" and not self._phase_executing_logged:
                    # Worker ready with specs - show "Executing on" before download
                    if self._waiting_start:
                        wait_sec = time.time() - self._waiting_start
                        logger.info("Wait time: %.0f sec", wait_sec)
                    worker_id = event_data.get("worker_id", "")
                    spec_parts = []
                    if event_data.get("storage_read_mbps"):
                        spec_parts.append(f"storage {event_data['storage_read_mbps']:.0f} MB/s")
                    if event_data.get("pcie_gbps"):
                        spec_parts.append(f"PCIe {event_data['pcie_gbps']:.1f} GB/s")
                    if event_data.get("network_mbps"):
                        spec_parts.append(f"net {event_data['network_mbps']:.0f} Mbps")
                    specs = ", ".join(spec_parts)
                    hw_str = self._initial_hw or ""
                    logger.info("Executing on %s: %s%s", worker_id, hw_str, f", {specs}" if specs else "")
                    self._phase_executing_logged = True
                elif name == "download_started" and not self._phase_downloading_logged:
                    self._phase_downloading_logged = True
                    size_mb = event_data.get("size_mb")
                    eta_sec = event_data.get("eta_sec")
                    size_str = f" ({size_mb:.0f} MB)" if size_mb else ""
                    eta_str = f", ETA ~{eta_sec:.0f}s" if eta_sec else ""
                    logger.info("Downloading assets%s%s", size_str, eta_str)
                elif name == "download_complete":
                    elapsed = event_data.get("elapsed_sec")
                    mbps = event_data.get("actual_mbps")
                    elapsed_str = f" in {elapsed:.0f}s" if elapsed else ""
                    mbps_str = f" ({mbps:.1f} MB/s)" if mbps else ""
                    logger.info("Download complete%s%s", elapsed_str, mbps_str)
                elif name and name != "stream_ended":
                    logger.debug("[event] %s", name)
            _user_on_log(msg)

        async with httpx.AsyncClient(timeout=30.0) as session:
            while True:
                data = await self._poll(session)

                # Log status transitions for user visibility (3 phases).
                current_status = data.get("status", "")
                if current_status and current_status not in TERMINAL_STATUSES:
                    hi = data.get("host_info") or {}
                    gpu = hi.get("gpu_model", "")
                    vram = hi.get("vram_gb", 0)
                    provider = hi.get("provider_name", "")
                    parts = [p for p in [gpu, f"{vram}GB" if vram else "", f"({provider})" if provider else ""] if p]
                    hw = ", ".join(parts)

                    if not self._phase_waiting_logged:
                        logger.info("Waiting for worker...")
                        self._phase_waiting_logged = True
                        self._waiting_start = time.time()

                    if not self._phase_host_logged and hw:
                        logger.info("Host obtained: %s", hw)
                        logger.info("Waiting for provision (up to 2 min)")
                        self._phase_host_logged = True
                        self._initial_hw = hw

                    if (
                        self._phase_host_logged
                        and not self._phase_provisioning_logged
                        and current_status == "provisioning"
                    ):
                        logger.info("Provisioning (up to 3 min)...")
                        self._phase_provisioning_logged = True

                    # Detect host change (e.g. first host failed during provisioning)
                    if (
                        self._initial_hw
                        and hw
                        and hw != self._initial_hw
                        and not self._host_change_logged
                    ):
                        logger.info("Host unavailable, reassigned to: %s", hw)
                        self._host_change_logged = True
                        self._initial_hw = hw  # update to avoid repeated logs

                    # Fallback: show "Executing on" by status if worker_ready event wasn't received
                    if current_status == "executing" and not self._phase_executing_logged:
                        if self._waiting_start:
                            wait_sec = time.time() - self._waiting_start
                            logger.info("Wait time: %.0f sec", wait_sec)
                        ws = data.get("worker_specs") or {}
                        spec_parts = []
                        if ws.get("storage_read_mbps"):
                            spec_parts.append(f"storage {ws['storage_read_mbps']:.0f} MB/s")
                        if ws.get("pcie_gbps"):
                            spec_parts.append(f"PCIe {ws['pcie_gbps']:.1f} GB/s")
                        if ws.get("network_mbps"):
                            spec_parts.append(f"net {ws['network_mbps']:.0f} Mbps")
                        specs = ", ".join(spec_parts)
                        worker_id = data.get("worker_id", "")
                        logger.info(
                            "Executing on %s: %s%s",
                            worker_id, hw, f", {specs}" if specs else "",
                        )
                        self._phase_executing_logged = True
                if current_status != self._last_status:
                    self._last_status = current_status

                # Start relay streaming on first poll that returns relay info.
                # Also retry if the relay task exited (connection error, auth fail, etc.)
                relay_dead = relay_task is not None and relay_task.done()
                if (
                    (relay_task is None or relay_dead)
                    and needs_relay
                    and data["status"] not in TERMINAL_STATUSES
                ):
                    relay_url_val = data.get("relay_url")
                    relay_token_val = data.get("relay_task_token")
                    if relay_url_val and relay_token_val:
                        relay_task = asyncio.create_task(
                            _relay_stream(
                                task_id=self.task_id,
                                relay_url=relay_url_val,
                                token=relay_token_val,
                                on_log=_on_log_effective,
                                ek_priv=self._ek_priv,
                                plaintext_code=self._plaintext_code,
                                plaintext_args=self._plaintext_args,
                                expected_worker_pub=data.get("worker_pub_b64"),
                                ca_pem=data.get("relay_ca"),
                            ),
                            name=f"relay-{self.task_id[:8]}",
                        )

                if data["status"] == "no_capacity":
                    if relay_task is not None:
                        relay_task.cancel()
                        relay_task = None
                    if self._resubmit is None:
                        raise NoCapacityError(
                            task_id=self.task_id,
                            message=data.get("message", ""),
                        )
                    self._no_capacity_attempts += 1
                    msg = data.get("message", "") or "no matching hosts"
                    logger.debug(
                        "No capacity (attempt %d, task_id=%s): %s — retrying in 5s",
                        self._no_capacity_attempts, self.task_id, msg,
                    )
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TaskTimeout(self.task_id, timeout)
                    await asyncio.sleep(min(5.0, remaining))
                    if loop.time() >= deadline:
                        raise TaskTimeout(self.task_id, timeout)
                    new_task_id = await self._resubmit()
                    self.task_id = new_task_id
                    self._last_status = ""
                    self._phase_waiting_logged = False
                    self._phase_host_logged = False
                    self._phase_provisioning_logged = False
                    self._phase_executing_logged = False
                    self._phase_downloading_logged = False
                    self._initial_hw = ""
                    self._host_change_logged = False
                    self._waiting_start = None
                    delay = 0.5
                    continue

                if data["status"] in TERMINAL_STATUSES:
                    if relay_task is not None:
                        # Wait briefly for relay to drain final messages
                        try:
                            await asyncio.wait_for(asyncio.shield(relay_task), timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                            relay_task.cancel()

                    # Race: status flips to terminal in Redis before CH writer
                    # flushes v2 with execution_result. For completed tasks we
                    # also retry until execution_result is populated; for
                    # failed/timeout/preempted output may legitimately be empty.
                    def _result_ready(d: dict) -> bool:
                        if d.get("result") is None:
                            return False
                        if d.get("status") != "completed":
                            return True
                        return bool((d.get("result") or {}).get("execution_result"))

                    if not _result_ready(data):
                        for _ in range(10):
                            await asyncio.sleep(0.5)
                            data = await self._poll(session)
                            if _result_ready(data):
                                break

                    if data.get("result") is not None:
                        self._result = TaskResult.from_response(data)
                    else:
                        self._result = TaskResult(
                            task_id=self.task_id, status=data["status"],
                        )

                    _log_billing_summary(self._result)
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
        for attempt in range(4):
            resp = await session.get(
                f"{self._client.broker_url}/tasks/{self.task_id}",
                headers={"X-API-Key": self._client.api_key},
            )
            if resp.status_code != 404 or attempt == 3:
                _check_response(resp)
                return resp.json()
            await asyncio.sleep(0.1)

    @staticmethod
    def _check_result(result: TaskResult) -> TaskResult:
        """Raise TaskError for non-completed results, return otherwise."""
        if result.status == "completed":
            return result

        if result.status == "timeout":
            raise RemoteTimeout(task_id=result.task_id)

        if result.status == "failed":
            if result.stderr == "e2e_payload_timeout":
                raise PayloadDeliveryError(task_id=result.task_id)
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
