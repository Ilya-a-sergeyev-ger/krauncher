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

from . import _inflight
from .exceptions import AuthError, E2EIdentityMismatch, InsufficientBalanceError, KrauncherError, NoCapacityError, PayloadDeliveryError, RemoteTimeout, RetriesExhausted, TaskError, TaskTimeout, ValueTransferError

if TYPE_CHECKING:
    from .KrauncherClient import KrauncherClient

TERMINAL_STATUSES = frozenset({"completed", "failed", "timeout", "hardware_preempted", "no_capacity", "aborted_insufficient_balance", "cancelled"})

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


@dataclass
class TaskGroup:
    """Shared-requirements envelope for tasks co-located on one worker.

    Built by :meth:`KrauncherClient.group` from the member tasks: the VRAM
    floor is the max over members, gpu/provider pins must not conflict, and
    the disk envelope covers every member's data — so the worker the first
    task provisions (Tier-1 group affinity pins without re-checking
    requirements) satisfies the whole group.

    Submit members with :meth:`submit`, or pass ``group=`` to
    :meth:`KrauncherClient.run_code` for code blocks.
    """

    group_id: str
    client: Any
    vram_floor: int = 0
    gpu_name: str | None = None
    gpu_arch: str | None = None
    provider: str | None = None
    disk_gb: int = 10

    async def submit(self, task: Callable, **kwargs: Any) -> "TaskHandle":
        """Submit a ``@client.task``-decorated member with the group envelope."""
        opts = getattr(task, "_krauncher_options", None)
        if opts is None:
            raise KrauncherError(
                "group.submit() expects a @client.task-decorated function"
            )
        # files= is the call-time channel for sending files, not a task
        # argument — same as the decorated wrapper does.
        files = kwargs.pop("files", None)
        return await self.client._submit(
            task._krauncher_code,
            task._krauncher_entry_point,
            kwargs,
            func_defaults=task._krauncher_defaults,
            classification_cache=task._krauncher_cls_cache,
            group=self,
            files=files,
            **opts,
        )


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
    # Files the task wrote in its working directory, {relative name: bytes}.
    # None = the task declared none, or the worker did not act on the
    # declaration; {} = handled, the task wrote no files.
    artifacts: dict[str, bytes] | None = None

    @property
    def files(self) -> list[str]:
        """Names of the artifacts the task produced, sorted."""
        return sorted(self.artifacts or {})

    def download(self, dest: str = ".") -> int:
        """Write the artifacts under *dest*, keeping their relative paths.

        Names arrive from the worker, so they are resolved and checked to stay
        under *dest* — a result must not be able to write anywhere on the
        caller's disk.

        Returns the number of files written.
        """
        from pathlib import Path

        if not self.artifacts:
            return 0
        root = Path(dest).resolve()
        root.mkdir(parents=True, exist_ok=True)
        for name, blob in self.artifacts.items():
            path = (root / name).resolve()
            if path == root or root not in path.parents:
                raise KrauncherError(
                    f"artifact name {name!r} escapes the download directory"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        return len(self.artifacts)

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
    # 24 MB both ways — the 16 MB plaintext inline budget after base64 + JSON
    # framing (payload out, FetchResult response in); python-grpc default
    # receive limit is 4 MB.
    options: list = [
        ("grpc.max_receive_message_length", 24 * 1024 * 1024),
        ("grpc.max_send_message_length", 24 * 1024 * 1024),
    ]
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
        if authority:
            options.append(("grpc.ssl_target_name_override", authority))
        creds = grpc.ssl_channel_credentials(root_certificates=ca_bytes)
        return grpc.secure_channel(target, creds, options=options)
    return grpc.insecure_channel(target, options=options)


def _relay_stream_sync(
    task_id: str,
    target: str,
    token: str,
    on_log: Callable[[dict[str, Any]], None],
    *,
    ek_priv: Any = None,
    plaintext_code: str | None = None,
    plaintext_args: dict[str, Any] | None = None,
    plaintext_artifacts: bool = False,
    plaintext_files: dict[str, bytes] | None = None,
    plaintext_credentials: dict[str, dict[str, str]] | None = None,
    expected_worker_pub: str | None = None,
    use_tls: bool = False,
    ca_pem: str | None = None,
    channel_holder: dict | None = None,
    key_holder: dict | None = None,
) -> None:
    """Blocking gRPC relay stream — meant to run in a worker thread.

    *channel_holder* (if given) receives the live gRPC channel under key
    ``"channel"`` so the async wrapper can close it on cancellation — this
    unblocks the iterator when the host dies abnormally and the relay never
    sends ``stream_ended`` (otherwise the executor thread blocks forever and
    hangs process exit).
    """
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
            if channel_holder is not None:
                channel_holder["channel"] = channel
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
                        if key_holder is not None:
                            # Retained for FetchResult decryption after the task ends.
                            key_holder["key"] = shared_key
                        logger.debug("[relay] shared key derived task_id=%s", task_id[:8])

                        payload_body: dict[str, Any] = {
                            "code_string": plaintext_code or "",
                            "args": plaintext_args or {},
                        }
                        # Storage credentials are the user's keys to third-party
                        # resources. Krauncher stores none of them: they are read
                        # from the caller's environment and ride this channel to
                        # the worker, never touching the broker.
                        if plaintext_credentials:
                            payload_body["credentials"] = plaintext_credentials
                        # Artifacts are data plane: the mount path, the
                        # transport and the produced file names are the user's,
                        # so they travel encrypted to the worker and never
                        # through the broker's task record.
                        if plaintext_artifacts:
                            payload_body["artifacts"] = True
                        payload_plain = _frame_payload(payload_body, plaintext_files)
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


def _frame_payload(body: dict[str, Any], files: dict[str, bytes] | None) -> bytes:
    """Serialize the task payload, carrying input files as raw bytes.

    Mirror of the worker's result frame: without files the wire form is the
    plain JSON object it has always been; with them, a JSON header, a newline,
    then the file bodies concatenated in header order. Files ride here rather
    than through storage because they are task data on the same client → relay
    → worker path as the code, which the broker never sees.
    """
    if not files:
        return json.dumps(body).encode()
    names = sorted(files)
    header = {**body, "files": [{"name": n, "size": len(files[n])} for n in names]}
    return json.dumps(header).encode() + b"\n" + b"".join(files[n] for n in names)


def _unframe_result(plaintext: bytes) -> dict[str, Any]:
    """Decode a task result blob into ``{"output": ..., "artifacts": {...}}``.

    A result without artifacts is a plain JSON object, exactly as before. With
    them the blob is framed: a JSON header, a newline, then the file bodies
    concatenated in header order — bytes are carried raw rather than base64'd
    into the JSON, which would cost a third of the size budget twice over.
    """
    split = plaintext.find(b"\n")
    header = json.loads(plaintext[:split] if split != -1 else plaintext)
    if "artifacts" not in header:
        return header

    body = plaintext[split + 1:] if split != -1 else b""
    declared = sum(entry["size"] for entry in header["artifacts"])
    if declared != len(body):
        raise ValueTransferError(
            f"artifact frame is inconsistent: header declares {declared} bytes, "
            f"body carries {len(body)}"
        )
    files: dict[str, bytes] = {}
    offset = 0
    for entry in header["artifacts"]:
        size = entry["size"]
        files[entry["name"]] = body[offset:offset + size]
        offset += size
    return {"output": header.get("output"), "artifacts": files}


def _fetch_relay_result_sync(
    task_id: str,
    target: str,
    token: str,
    use_tls: bool,
    ca_pem: str | None,
    shared_key: bytes,
) -> dict[str, Any] | None:
    """Fetch and decrypt the task's result envelope from the relay mailbox.

    Returns the ``{"output": ...}`` envelope, or None when the mailbox has
    nothing stored (NotFound) or the fetch/decrypt failed. Blocking — run in
    the relay thread pool.
    """
    try:
        import grpc
        from . import relay_pb2, relay_pb2_grpc
        from .crypto import decrypt
    except ImportError as exc:
        logger.debug("[relay] result fetch import error: %s", exc)
        return None

    try:
        with _make_relay_channel(target, use_tls, ca_pem) as channel:
            stub = relay_pb2_grpc.RelayStub(channel)
            resp = stub.FetchResult(
                relay_pb2.FetchResultRequest(task_id=task_id),
                metadata=[("authorization", f"bearer {token}")],
                timeout=30.0,
            )
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.NOT_FOUND:
            logger.debug("[relay] no stored result task_id=%s", task_id[:8])
        else:
            logger.warning("[relay] result fetch failed task_id=%s: %s", task_id[:8], exc)
        return None
    except Exception as exc:
        logger.warning("[relay] result fetch failed task_id=%s: %s", task_id[:8], exc)
        return None

    try:
        envelope = json.loads(resp.data)
        plaintext = decrypt(shared_key, envelope["enc"])
        return _unframe_result(plaintext)
    except Exception as exc:
        logger.warning("[relay] result decrypt failed task_id=%s: %s", task_id[:8], exc)
        return None


async def _relay_stream(
    task_id: str,
    relay_url: str,
    token: str,
    on_log: Callable[[dict[str, Any]], None],
    *,
    ek_priv: Any = None,
    plaintext_code: str | None = None,
    plaintext_args: dict[str, Any] | None = None,
    plaintext_artifacts: bool = False,
    plaintext_files: dict[str, bytes] | None = None,
    plaintext_credentials: dict[str, dict[str, str]] | None = None,
    expected_worker_pub: str | None = None,
    ca_pem: str | None = None,
    key_holder: dict | None = None,
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
    channel_holder: dict = {}
    try:
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
                plaintext_artifacts=plaintext_artifacts,
                plaintext_files=plaintext_files,
                plaintext_credentials=plaintext_credentials,
                expected_worker_pub=expected_worker_pub,
                use_tls=use_tls,
                ca_pem=ca_pem,
                channel_holder=channel_holder,
                key_holder=key_holder,
            ),
        )
    except asyncio.CancelledError:
        # Cancelled on terminal status / timeout. The executor thread may be
        # blocked on a stream that the relay never closes (abnormal host death);
        # close the channel so the iterator raises and the thread exits, else
        # the non-daemon executor thread hangs process exit.
        ch = channel_holder.get("channel")
        if ch is not None:
            try:
                ch.close()
            except Exception:
                pass
        raise


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
        resubmit: Callable[..., Any] | None = None,
        stream_stderr: bool = False,
        artifacts: bool = False,
        files: dict[str, bytes] | None = None,
        credentials: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.task_id = task_id
        self._artifacts = artifacts
        self._files = files
        self._client = client
        self._result: TaskResult | None = None
        self._last_status: str = ""
        # Relay coords (url/token/ca) for cancel-on-abandon of an executing task
        # — calling relay CancelTask kills the worker container so it emits a
        # TaskResult and the broker settles + releases the hold. Set on poll.
        self._relay_cancel_info: dict | None = None
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
        # E2E fields (E2E is mandatory)
        self._ek_priv = ek_priv
        self._plaintext_code = plaintext_code
        self._plaintext_args = plaintext_args
        self._credentials = credentials
        # Shared task key captured by the relay stream thread; reused to
        # decrypt the FetchResult envelope after the task ends.
        self._e2e_key_holder: dict = {}
        #: Task classification result (TaskClassification or None)
        self.classification = classification
        # Re-submit callback: returns a new task_id when invoked. Used by wait()
        # to transparently retry a failure the broker flagged as retriable. If
        # None, the failure is raised to the caller so they can implement their
        # own policy.
        self._resubmit = resubmit
        self._retry_attempts: int = 0
        # True once this attempt reached "executing": from then on the clock
        # belongs to the task, so a deadline hit is a real timeout, not ours.
        self._reached_executing: bool = False
        #: When true and wait() is called without on_log, install a default handler
        #: that mirrors remote stdout/stderr to local sys.stdout/sys.stderr.
        self.stream_stderr = stream_stderr

    def __repr__(self) -> str:
        return f"TaskHandle(task_id={self.task_id!r})"

    def _reset_for_retry(self) -> None:
        """Clear per-attempt state so the new task_id reports its phases afresh."""
        self._last_status = ""
        self._reached_executing = False
        self._phase_waiting_logged = False
        self._phase_host_logged = False
        self._phase_provisioning_logged = False
        self._phase_executing_logged = False
        self._phase_downloading_logged = False
        self._initial_hw = ""
        self._host_change_logged = False
        self._waiting_start = None

    def __await__(self):
        """Allow ``result = await task``."""
        return self.wait().__await__()

    def _cancel_remote(self, reason: str = "user") -> None:
        """Best-effort synchronous DELETE /tasks/{id} so the broker releases the
        prepaid hold when the caller abandons the task before it reaches a
        terminal status (Ctrl-C, timeout, exception).

        ``reason`` tells the broker who willed the cancel: ``user`` (the
        caller gave up) or ``infra_retry`` (we are cancelling a task that
        never started, to submit it again). An ``infra_retry`` cancel is
        settled as an infrastructure fault — no dispatch fee.

        Synchronous on purpose: it must complete even while the event loop is
        being torn down by Ctrl-C, where an awaited call could be re-cancelled.
        """
        # 1. Tell the broker. Queued/pre-dispatch → settle fee + release hold.
        #    Executing → records cancel intent (final settle via the worker's
        #    TaskResult triggered by step 2).
        try:
            import httpx as _httpx
            with _httpx.Client(timeout=10.0) as s:
                s.delete(
                    f"{self._client.broker_url}/tasks/{self.task_id}",
                    headers={"X-API-Key": self._client.api_key},
                    params={"reason": reason},
                )
            logger.debug("cancel-on-abandon sent: %s", self.task_id)
        except Exception as e:
            logger.debug("cancel-on-abandon failed for %s: %s", self.task_id, e)

        # 2. If the task is executing, actively stop the worker via relay
        #    CancelTask (kills the container → worker emits TaskResult → broker
        #    settles + releases the hold). The broker can't do this (CancelTask
        #    is client-authed); the client holds the per-task relay token.
        info = self._relay_cancel_info
        if info:
            self._cancel_via_relay(info)

        _inflight.unregister(self)

    def _cancel_via_relay(self, info: dict) -> None:
        """Best-effort synchronous relay CancelTask to stop an executing worker."""
        try:
            import grpc  # noqa: F401
            from . import relay_pb2, relay_pb2_grpc
        except ImportError:
            return
        try:
            url = info["url"]
            token = info["token"]
            use_tls = _detect_relay_tls(url)
            target = url
            for prefix in _TLS_SCHEMES + _PLAINTEXT_SCHEMES:
                if target.startswith(prefix):
                    target = target[len(prefix):]
                    break
            target = target.rstrip("/")
            with _make_relay_channel(target, use_tls, info.get("ca")) as channel:
                stub = relay_pb2_grpc.RelayStub(channel)
                stub.CancelTask(
                    relay_pb2.CancelTaskRequest(task_id=self.task_id),
                    metadata=[("authorization", f"bearer {token}")],
                    timeout=10.0,
                )
            logger.debug("relay cancel sent: %s", self.task_id)
        except Exception as e:
            logger.debug("relay cancel failed for %s: %s", self.task_id, e)

    async def wait(
        self,
        *,
        timeout: float = 600.0,
        on_log: Callable[[dict[str, Any]], None] | None = None,
    ) -> TaskResult:
        """Wait for terminal status; on abandonment before terminal (Ctrl-C,
        timeout, exception) cancel the task remotely so its hold is released."""
        try:
            result = await self._wait_impl(timeout=timeout, on_log=on_log)
            _inflight.unregister(self)
            return result
        except (asyncio.CancelledError, KeyboardInterrupt, Exception):
            # Exited without a terminal result → the caller abandoned the task.
            # Tell the broker so the prepaid hold is freed instead of leaking.
            if self._result is None:
                self._cancel_remote()
            raise

    async def _wait_impl(
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

        E2E is always on: the relay stream is used to perform the key exchange
        and deliver the encrypted task payload to the worker.
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
        # Ceiling on the whole chain of attempts, fixed at the first submit:
        # retries must not keep the caller waiting without bound.
        chain_budget = self._client.max_task_chain_sec or (2 * timeout)
        chain_deadline = loop.time() + chain_budget
        delay = 0.5
        relay_task: asyncio.Task | None = None

        # When stream_stderr is on and the caller didn't provide on_log, mirror
        # remote stdout/stderr to local sys.stdout/sys.stderr and render GPU
        # metrics as a single \r-overwritten progress line.
        if on_log is None and self.stream_stderr:
            import sys as _sys
            import time as _t

            _pl = {"start": None, "open_len": 0}

            def _close_progress_line() -> None:
                if _pl["open_len"]:
                    _sys.stdout.write("\r" + " " * _pl["open_len"] + "\r")
                    _sys.stdout.flush()
                    _pl["open_len"] = 0

            def _console_on_log(msg: dict) -> None:
                t = msg.get("type")
                d = msg.get("data") or {}
                if t == "stdout":
                    _close_progress_line()
                    _sys.stdout.write(d.get("text", ""))
                    _sys.stdout.flush()
                elif t == "stderr":
                    _close_progress_line()
                    _sys.stderr.write(d.get("text", ""))
                    _sys.stderr.flush()
                elif t == "metric":
                    if _pl["start"] is None:
                        _pl["start"] = _t.monotonic()
                    line = (
                        f"[gpu {d.get('gpu_util_pct', 0):3.0f}% · "
                        f"vram {d.get('vram_used_gb', 0):.1f}/"
                        f"{d.get('vram_total_gb', 0):.0f} GB · "
                        f"{_t.monotonic() - _pl['start']:.0f}s]"
                    )
                    pad = max(0, _pl["open_len"] - len(line))
                    _sys.stdout.write("\r" + line + " " * pad)
                    _sys.stdout.flush()
                    _pl["open_len"] = len(line)
                elif t == "event":
                    name = d.get("name", "")
                    if name == "execution_started":
                        _pl["start"] = _t.monotonic()
                    elif name in ("execution_complete", "stream_ended"):
                        _close_progress_line()

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
                        spec_parts.append(f"net {event_data['network_mbps']:.0f} MB/s")
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

                # Stash relay coords so cancel-on-abandon can actively stop an
                # executing worker via relay CancelTask (not just flag it).
                _rurl = data.get("relay_url")
                _rtok = data.get("relay_task_token")
                if _rurl and _rtok:
                    self._relay_cancel_info = {
                        "url": _rurl, "token": _rtok, "ca": data.get("relay_ca"),
                    }

                # Log status transitions for user visibility (3 phases).
                current_status = data.get("status", "")
                if current_status == "executing":
                    self._reached_executing = True
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
                            spec_parts.append(f"net {ws['network_mbps']:.0f} MB/s")
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
                                plaintext_artifacts=self._artifacts,
                                plaintext_files=self._files,
                                plaintext_credentials=self._credentials,
                                expected_worker_pub=data.get("worker_pub_b64"),
                                ca_pem=data.get("relay_ca"),
                                key_holder=self._e2e_key_holder,
                            ),
                            name=f"relay-{self.task_id[:8]}",
                        )

                # Infrastructure fault: the broker flags it retriable (stage 1).
                # ``no_capacity`` is treated as retriable even without the flag,
                # so an older broker keeps its previous behaviour.
                status_now = data["status"]
                if (
                    status_now in TERMINAL_STATUSES
                    and (data.get("retriable") or status_now == "no_capacity")
                ):
                    if relay_task is not None:
                        relay_task.cancel()
                        relay_task = None
                    msg = data.get("message", "") or status_now
                    max_retries = self._client.max_task_retries

                    if (
                        self._resubmit is not None
                        and self._retry_attempts < max_retries
                        and loop.time() < chain_deadline
                    ):
                        self._retry_attempts += 1
                        logger.info(
                            "Infrastructure failure (%s), retrying %d/%d: %s",
                            status_now, self._retry_attempts, max_retries, msg,
                        )
                        await asyncio.sleep(5.0)
                        self.task_id = await self._resubmit(
                            self.task_id, self._retry_attempts + 1,
                        )
                        self._reset_for_retry()
                        # Each attempt gets the full timeout again, but never
                        # past the chain's wall-clock ceiling.
                        deadline = min(loop.time() + timeout, chain_deadline)
                        delay = 0.5
                        continue

                    # Out of attempts (or no resubmit callback at all).
                    if status_now == "no_capacity":
                        raise NoCapacityError(
                            task_id=self.task_id,
                            message=data.get("message", ""),
                        )
                    if self._retry_attempts:
                        raise RetriesExhausted(
                            task_id=self.task_id, status=status_now,
                            attempts=self._retry_attempts,
                            message=data.get("message", ""),
                        )
                    # Never retried (no callback) — fall through to the normal
                    # terminal handling so the caller sees the usual result.

                if data["status"] in TERMINAL_STATUSES:
                    if relay_task is not None:
                        # Wait briefly for relay to drain final messages
                        try:
                            await asyncio.wait_for(asyncio.shield(relay_task), timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                            # Drain timed out (abnormal host death — relay never
                            # closes the stream). Cancel and await so the channel
                            # is closed and the executor thread exits before we
                            # return; otherwise the non-daemon thread hangs exit.
                            relay_task.cancel()
                            try:
                                await relay_task
                            except (asyncio.CancelledError, Exception):
                                pass

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

                    # E2E: the output travels via the relay result mailbox,
                    # not the broker (which stores output=None for E2E tasks).
                    if self._ek_priv is not None and data["status"] == "completed":
                        await self._merge_relay_result(data)

                    self._check_artifacts_delivered()
                    _log_billing_summary(self._result)
                    return self._check_result(self._result)

                remaining = deadline - loop.time()
                if remaining <= 0:
                    if relay_task is not None:
                        relay_task.cancel()
                        relay_task = None
                    # A deadline hit before the task ever reached "executing"
                    # is our infrastructure being slow (provisioning, queue),
                    # not the task overrunning — cancel it and start a new one.
                    # Once it is executing the clock is the task's own, and an
                    # overrun is a real timeout: no retry.
                    if (
                        not self._reached_executing
                        and self._resubmit is not None
                        and self._retry_attempts < self._client.max_task_retries
                        and loop.time() < chain_deadline
                    ):
                        self._retry_attempts += 1
                        logger.info(
                            "Task never started within %.0fs (still %s), "
                            "retrying %d/%d",
                            timeout, self._last_status or "queued",
                            self._retry_attempts, self._client.max_task_retries,
                        )
                        self._cancel_remote("infra_retry")
                        self.task_id = await self._resubmit(
                            self.task_id, self._retry_attempts + 1,
                        )
                        self._reset_for_retry()
                        deadline = min(loop.time() + timeout, chain_deadline)
                        delay = 0.5
                        continue
                    # A chain that ran out of wall clock never delivered a
                    # result either, but the cause is ours, not the task's.
                    if self._retry_attempts and loop.time() >= chain_deadline:
                        self._cancel_remote("infra_retry")
                        raise RetriesExhausted(
                            task_id=self.task_id,
                            status=self._last_status or "queued",
                            attempts=self._retry_attempts,
                            message=(
                                f"chain wall-clock budget of {chain_budget:.0f}s exhausted"
                            ),
                        )
                    raise TaskTimeout(self.task_id, timeout)

                await asyncio.sleep(min(delay, remaining))
                delay = min(delay * 1.5, 5.0)


    async def _merge_relay_result(self, data: dict[str, Any]) -> None:
        """Fetch the encrypted output from the relay mailbox into the result.

        Legacy compatibility: when the mailbox has nothing but the broker
        record carries an output (worker not yet upgraded), the broker value
        stays. A completed E2E task with neither raises TaskError — by design
        there is no broker-side fallback for E2E results.
        """
        cancel_info = self._relay_cancel_info or {}
        relay_url = data.get("relay_url") or cancel_info.get("url")
        token = data.get("relay_task_token") or cancel_info.get("token")
        ca_pem = data.get("relay_ca") or cancel_info.get("ca")

        # Shared key: captured during the stream's key exchange, or re-derived
        # from the broker-reported worker pubkey (covers stream races).
        key = self._e2e_key_holder.get("key")
        if key is None and data.get("worker_pub_b64"):
            try:
                import base64
                from .crypto import derive_shared_secret
                key = derive_shared_secret(
                    self._ek_priv,
                    base64.urlsafe_b64decode(data["worker_pub_b64"] + "=="),
                )
            except Exception as exc:
                logger.debug("shared key derivation failed: %s", exc)

        envelope: dict[str, Any] | None = None
        if relay_url and token and key is not None:
            use_tls = _detect_relay_tls(relay_url)
            target = relay_url
            for prefix in _TLS_SCHEMES + _PLAINTEXT_SCHEMES:
                if target.startswith(prefix):
                    target = target[len(prefix):]
                    break
            target = target.rstrip("/")
            loop = asyncio.get_event_loop()
            envelope = await loop.run_in_executor(
                _relay_executor,
                lambda: _fetch_relay_result_sync(
                    self.task_id, target, token, use_tls, ca_pem, key,
                ),
            )

        if envelope is not None:
            import dataclasses
            self._result = dataclasses.replace(
                self._result,
                output=envelope.get("output"),
                artifacts=envelope.get("artifacts"),
            )
        elif self._result.output is None:
            raise TaskError(
                "Task completed but its result was not delivered from the "
                "relay mailbox (retention expired, storage overflow, or the "
                "worker could not upload it). Re-run the task.",
                task_id=self.task_id,
            )

    def _check_artifacts_delivered(self) -> None:
        """Fail loudly when a declared artifact set never came back.

        The manifest distinguishes "the task wrote no files" (an empty list)
        from "nothing acted on the declaration" (absent). Without this check
        an unsupported broker or worker would drop ``artifacts=`` silently and
        the caller would be left looking for files that were never collected.
        """
        if not self._artifacts or self._result is None:
            return
        if self._result.status != "completed":
            return
        if self._result.artifacts is not None:
            return
        raise KrauncherError(
            f"task {self.task_id} declared artifacts but none were reported "
            f"back. The worker that ran it does not support the artifacts API "
            f"yet — write the files to an output data source until it does."
        )

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

        if result.status == "aborted_insufficient_balance":
            raise TaskError(
                f"Task {result.task_id} aborted: insufficient balance for the "
                f"next billing chunk (top up and re-submit)",
                task_id=result.task_id,
            )

        if result.status == "cancelled":
            raise TaskError(
                f"Task {result.task_id} was cancelled",
                task_id=result.task_id,
            )

        raise TaskError(
            f"Task {result.task_id} was preempted ({result.status})",
            task_id=result.task_id,
            remote_traceback=None,
        )
