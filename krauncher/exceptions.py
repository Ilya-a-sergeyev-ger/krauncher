# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Exceptions for krauncher client library."""

from __future__ import annotations


class KrauncherError(Exception):
    """Base exception for all krauncher errors."""


class AuthError(KrauncherError):
    """Raised on 401/403 from the broker."""


class InsufficientBalanceError(KrauncherError):
    """Raised on HTTP 402 from POST /tasks when the user's available KU
    balance is below the predicted hold amount.

    The broker responds with a structured detail::

        {
            "error": "insufficient_balance",
            "required_ku":  <hold_ku>,
            "predicted_ku": <predicted_ku>,
            "fee_ku":       <fee_ku>,
            "available_ku": <balance_ku - held_ku>,
            "balance_ku":   <balance_ku>,
            "held_ku":      <held_ku>,
        }

    These fields are exposed as attributes; missing fields default to 0.0.
    """

    def __init__(self, detail: dict | str) -> None:
        if isinstance(detail, dict):
            self.required_ku = float(detail.get("required_ku", 0.0) or 0.0)
            self.predicted_ku = float(detail.get("predicted_ku", 0.0) or 0.0)
            self.fee_ku = float(detail.get("fee_ku", 0.0) or 0.0)
            self.available_ku = float(detail.get("available_ku", 0.0) or 0.0)
            self.balance_ku = float(detail.get("balance_ku", 0.0) or 0.0)
            self.held_ku = float(detail.get("held_ku", 0.0) or 0.0)
            msg = (
                f"Insufficient balance: need {self.required_ku:.4f} KU "
                f"(predicted {self.predicted_ku:.4f} + fee {self.fee_ku:.4f}), "
                f"available {self.available_ku:.4f} KU "
                f"(balance {self.balance_ku:.4f}, held {self.held_ku:.4f})"
            )
        else:
            self.required_ku = 0.0
            self.predicted_ku = 0.0
            self.fee_ku = 0.0
            self.available_ku = 0.0
            self.balance_ku = 0.0
            self.held_ku = 0.0
            msg = f"Insufficient balance: {detail}"
        super().__init__(msg)


class TaskError(KrauncherError):
    """Raised when a task fails or is preempted on the worker side.

    The remote traceback is included in the exception message so that
    ``print(e)`` or an unhandled exception shows the full remote stack
    as if the error happened locally.
    """

    def __init__(
        self, message: str, *, task_id: str, remote_traceback: str | None = None,
    ) -> None:
        self.task_id = task_id
        self.remote_traceback = remote_traceback

        if remote_traceback:
            full = f"{message}\n\n--- Remote Traceback (task {task_id}) ---\n{remote_traceback}"
        else:
            full = message
        super().__init__(full)


class RemoteTimeout(TaskError):
    """Raised when the worker killed the task due to execution timeout.

    Unlike TaskTimeout (client-side polling timeout), this means the remote
    worker enforced the timeout limit and terminated the container.
    """

    def __init__(self, task_id: str, timeout_sec: float | None = None) -> None:
        self.timeout_sec = timeout_sec
        super().__init__(
            f"Task {task_id} killed by worker: execution timeout"
            + (f" ({timeout_sec}s)" if timeout_sec is not None else ""),
            task_id=task_id,
            remote_traceback=None,
        )


class PayloadDeliveryError(TaskError):
    """Raised when the encrypted payload could not be delivered to the worker.

    This typically means the client could not establish a relay connection
    in time (e.g. due to high concurrency or network issues).  The task
    was not executed and you were not charged.  Retrying usually helps.
    """

    def __init__(self, task_id: str) -> None:
        super().__init__(
            "Could not deliver task payload to the worker — the secure "
            "channel was not established in time. The task was not executed "
            "and no charges were applied. This can happen under heavy load; "
            "please retry.",
            task_id=task_id,
            remote_traceback=None,
        )


class TaskTimeout(KrauncherError):
    """Raised when TaskHandle.wait() exceeds its timeout."""

    def __init__(self, task_id: str, timeout: float) -> None:
        super().__init__(f"Task {task_id} did not complete within {timeout}s")
        self.task_id = task_id
        self.timeout = timeout


class NoCapacityError(KrauncherError):
    """Raised when the broker reports no matching hosts for the task.

    The task did not run and was not charged. The high-level submit/run
    wrappers catch this and retry on a fixed interval; raw TaskHandle.wait()
    surfaces it directly so callers using the low-level API can implement
    their own policy.
    """

    def __init__(self, task_id: str, message: str = "") -> None:
        super().__init__(
            f"No matching hosts available for task {task_id}"
            + (f": {message}" if message else "")
        )
        self.task_id = task_id
        self.broker_message = message


class RetriesExhausted(KrauncherError):
    """Raised when every transparent retry of an infrastructure failure failed.

    The task never ran to completion: each attempt died for a reason the
    broker flagged as our fault (host lost, no capacity, ...). ``status``
    carries the last one. Raise the retry budget via
    ``KrauncherClient(max_task_retries=...)`` / ``CAS_MAX_TASK_RETRIES``.
    """

    def __init__(self, task_id: str, status: str, attempts: int, message: str = "") -> None:
        super().__init__(
            f"Task {task_id} gave up after {attempts} infrastructure "
            f"retries (last status: {status})"
            + (f": {message}" if message else "")
        )
        self.task_id = task_id
        self.status = status
        self.attempts = attempts
        self.broker_message = message


class SerializationError(KrauncherError):
    """Raised when a function cannot be serialized for remote execution."""


class ValueTransferError(KrauncherError):
    """Raised when a value cannot cross the caller ↔ task boundary.

    Inputs must be JSON-serializable and fit the inline budget (see
    ``krauncher.values``); outputs must come back as the ``{name: value}``
    dict with every requested name present.
    """


class E2EIdentityMismatch(KrauncherError):
    """Raised when the worker pubkey from relay does not match what the
    broker reported. Indicates a relay attempting an MITM on the E2E channel.

    Only raised when ``KRAUNCHER_E2E_STRICT=1``; otherwise a warning is logged.
    """

    def __init__(self, task_id: str) -> None:
        super().__init__(
            f"E2E identity mismatch for task {task_id}: relay-supplied worker "
            "pubkey differs from broker-reported value (possible MITM)."
        )
        self.task_id = task_id
