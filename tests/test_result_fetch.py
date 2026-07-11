# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for the E2E relay result fetch merge (TaskHandle._merge_relay_result)."""

from unittest.mock import MagicMock, patch

import pytest

from krauncher import KrauncherClient, TaskError
from krauncher.models import TaskHandle, TaskResult


def _handle() -> TaskHandle:
    client = KrauncherClient(api_key="cas_test", broker_url="http://broker:8000")
    h = TaskHandle(task_id="t-1", client=client, ek_priv=object())
    h._e2e_key_holder["key"] = b"k" * 32
    h._relay_cancel_info = {"url": "relay:9001", "token": "tok", "ca": None}
    return h


def _completed(output) -> TaskResult:
    return TaskResult(task_id="t-1", status="completed", output=output)


DATA = {"status": "completed"}


@pytest.mark.asyncio
async def test_envelope_output_wins():
    h = _handle()
    h._result = _completed(None)
    with patch("krauncher.models._fetch_relay_result_sync",
               return_value={"output": {"preds": [1, 2]}}):
        await h._merge_relay_result(DATA)
    assert h._result.output == {"preds": [1, 2]}


@pytest.mark.asyncio
async def test_envelope_with_null_output_is_legit():
    # The worker uploads {"output": null} for tasks that return None — that is
    # a delivered result, not a loss.
    h = _handle()
    h._result = _completed(None)
    with patch("krauncher.models._fetch_relay_result_sync",
               return_value={"output": None}):
        await h._merge_relay_result(DATA)
    assert h._result.output is None


@pytest.mark.asyncio
async def test_mailbox_miss_keeps_legacy_broker_output():
    h = _handle()
    h._result = _completed({"legacy": True})
    with patch("krauncher.models._fetch_relay_result_sync", return_value=None):
        await h._merge_relay_result(DATA)
    assert h._result.output == {"legacy": True}


@pytest.mark.asyncio
async def test_mailbox_miss_without_output_raises():
    h = _handle()
    h._result = _completed(None)
    with patch("krauncher.models._fetch_relay_result_sync", return_value=None):
        with pytest.raises(TaskError, match="not delivered"):
            await h._merge_relay_result(DATA)


@pytest.mark.asyncio
async def test_no_relay_coords_without_output_raises():
    h = _handle()
    h._relay_cancel_info = None
    h._result = _completed(None)
    # No relay coords and no key derivable → fetch impossible → TaskError.
    with pytest.raises(TaskError, match="not delivered"):
        await h._merge_relay_result(DATA)
