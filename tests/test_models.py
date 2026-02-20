"""Tests for krauncher.models — TaskResult and TaskHandle."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from krauncher.exceptions import RemoteTimeout, TaskError, TaskTimeout
from krauncher.models import TaskHandle, TaskResult


# -- TaskResult ----------------------------------------------------------------


class TestTaskResult:
    def test_from_response_completed(self):
        data = {
            "task_id": "abc-123",
            "status": "completed",
            "result": {
                "task_id": "abc-123",
                "status": "completed",
                "system_info": {
                    "exit_code": 0,
                    "actual_gpu": "NVIDIA RTX 4090",
                    "execution_time_sec": 12.5,
                },
                "execution_result": {
                    "output": {"loss": 0.01},
                    "stdout": "training...\n",
                    "stderr": "",
                    "traceback": None,
                },
                "billing_metrics": {
                    "duration_sec": 15.0,
                    "gpu_util_avg": 85.0,
                    "price_per_hour_usd": 1.20,
                },
            },
        }
        r = TaskResult.from_response(data)
        assert r.task_id == "abc-123"
        assert r.status == "completed"
        assert r.output == {"loss": 0.01}
        assert r.stdout == "training...\n"
        assert r.actual_gpu == "NVIDIA RTX 4090"
        assert r.execution_time_sec == 12.5
        assert r.duration_sec == 15.0
        assert r.gpu_util_avg == 85.0
        assert r.cost_usd == pytest.approx(15.0 * 1.20 / 3600.0)

    def test_from_response_failed(self):
        data = {
            "task_id": "abc-456",
            "status": "failed",
            "result": {
                "system_info": {"exit_code": 1},
                "execution_result": {
                    "output": None,
                    "stderr": "RuntimeError: CUDA OOM",
                    "traceback": "Traceback...\nRuntimeError: CUDA OOM",
                },
                "billing_metrics": {},
            },
        }
        r = TaskResult.from_response(data)
        assert r.status == "failed"
        assert r.exit_code == 1
        assert "CUDA OOM" in r.stderr
        assert r.traceback is not None

    def test_from_response_no_result(self):
        data = {"task_id": "abc-789", "status": "completed", "result": None}
        r = TaskResult.from_response(data)
        assert r.output is None
        assert r.cost_usd == 0.0


# -- TaskHandle ----------------------------------------------------------------


def _make_handle(task_id: str = "test-task-1") -> TaskHandle:
    """Create a TaskHandle with a mock client."""
    mock_client = type("MockClient", (), {
        "api_key": "cas_test",
        "broker_url": "http://localhost:8000",
    })()
    return TaskHandle(task_id=task_id, client=mock_client)


class TestTaskHandle:
    @pytest.mark.asyncio
    async def test_wait_success(self):
        handle = _make_handle()

        responses = [
            {"task_id": "test-task-1", "status": "queued", "result": None},
            {"task_id": "test-task-1", "status": "executing", "result": None},
            {
                "task_id": "test-task-1",
                "status": "completed",
                "result": {
                    "system_info": {"exit_code": 0, "actual_gpu": "RTX 4090",
                                    "execution_time_sec": 5.0},
                    "execution_result": {"output": 42, "stdout": "", "stderr": "",
                                         "traceback": None},
                    "billing_metrics": {"duration_sec": 5.0, "gpu_util_avg": 50.0,
                                        "price_per_hour_usd": 1.0},
                },
            },
        ]

        call_idx = 0

        async def mock_poll(session):
            nonlocal call_idx
            data = responses[min(call_idx, len(responses) - 1)]
            call_idx += 1
            return data

        handle._poll = mock_poll

        result = await handle.wait(timeout=10.0)
        assert result.status == "completed"
        assert result.output == 42
        assert handle.done()

    @pytest.mark.asyncio
    async def test_wait_failure_raises_task_error(self):
        handle = _make_handle()

        async def mock_poll(session):
            return {
                "task_id": "test-task-1",
                "status": "failed",
                "result": {
                    "system_info": {"exit_code": 1},
                    "execution_result": {"output": None, "stderr": "boom",
                                         "traceback": "Traceback..."},
                    "billing_metrics": {},
                },
            }

        handle._poll = mock_poll

        with pytest.raises(TaskError, match="failed"):
            await handle.wait(timeout=5.0)

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        handle = _make_handle()

        async def mock_poll(session):
            return {"task_id": "test-task-1", "status": "executing", "result": None}

        handle._poll = mock_poll

        with pytest.raises(TaskTimeout):
            await handle.wait(timeout=0.5)

    @pytest.mark.asyncio
    async def test_await_syntax(self):
        """``await handle`` is equivalent to ``await handle.wait()``."""
        handle = _make_handle()

        async def mock_poll(session):
            return {
                "task_id": "test-task-1",
                "status": "completed",
                "result": {
                    "system_info": {},
                    "execution_result": {"output": "ok"},
                    "billing_metrics": {},
                },
            }

        handle._poll = mock_poll
        result = await handle
        assert result.output == "ok"

    @pytest.mark.asyncio
    async def test_preempted_raises_task_error(self):
        handle = _make_handle()

        async def mock_poll(session):
            return {
                "task_id": "test-task-1",
                "status": "hardware_preempted",
                "result": {
                    "system_info": {},
                    "execution_result": {},
                    "billing_metrics": {},
                },
            }

        handle._poll = mock_poll

        with pytest.raises(TaskError, match="preempted"):
            await handle.wait(timeout=5.0)

    def test_done_false_initially(self):
        handle = _make_handle()
        assert not handle.done()

    @pytest.mark.asyncio
    async def test_wait_timeout_status_raises_remote_timeout(self):
        handle = _make_handle()

        async def mock_poll(session):
            return {
                "task_id": "test-task-1",
                "status": "timeout",
                "result": {
                    "system_info": {"exit_code": -1},
                    "execution_result": {"output": None, "stderr": "timeout",
                                         "traceback": None},
                    "billing_metrics": {"duration_sec": 10.0},
                },
            }

        handle._poll = mock_poll

        with pytest.raises(RemoteTimeout):
            await handle.wait(timeout=5.0)

    @pytest.mark.asyncio
    async def test_remote_timeout_is_task_error(self):
        handle = _make_handle()

        async def mock_poll(session):
            return {
                "task_id": "test-task-1",
                "status": "timeout",
                "result": {
                    "system_info": {},
                    "execution_result": {},
                    "billing_metrics": {},
                },
            }

        handle._poll = mock_poll

        with pytest.raises(TaskError):
            await handle.wait(timeout=5.0)

    def test_from_response_timeout(self):
        data = {
            "task_id": "abc-timeout",
            "status": "timeout",
            "result": {
                "system_info": {"exit_code": -1, "execution_time_sec": 10.0},
                "execution_result": {"output": None, "stderr": "Container killed after 10s timeout"},
                "billing_metrics": {"duration_sec": 10.0, "price_per_hour_usd": 1.0},
            },
        }
        r = TaskResult.from_response(data)
        assert r.status == "timeout"
        assert r.exit_code == -1
        assert r.output is None

    def test_repr(self):
        handle = _make_handle("abc-123")
        assert "abc-123" in repr(handle)
