"""Exceptions for krauncher client library."""

from __future__ import annotations


class KrauncherError(Exception):
    """Base exception for all krauncher errors."""


class AuthError(KrauncherError):
    """Raised on 401/403 from the broker."""


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


class TaskTimeout(KrauncherError):
    """Raised when TaskHandle.wait() exceeds its timeout."""

    def __init__(self, task_id: str, timeout: float) -> None:
        super().__init__(f"Task {task_id} did not complete within {timeout}s")
        self.task_id = task_id
        self.timeout = timeout


class SerializationError(KrauncherError):
    """Raised when a function cannot be serialized for remote execution."""
