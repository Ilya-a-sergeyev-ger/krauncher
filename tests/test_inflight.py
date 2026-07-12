# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for the in-flight task registry (krauncher._inflight)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from krauncher import _inflight


@pytest.fixture(autouse=True)
def _clean_registry():
    with _inflight._lock:
        _inflight._inflight.clear()
    _inflight._hook_installed = False
    yield
    with _inflight._lock:
        _inflight._inflight.clear()
    _inflight._hook_installed = False


def _fake_handle(task_id="t-1", result=None, last_status=""):
    h = MagicMock()
    h.task_id = task_id
    h._result = result
    h._last_status = last_status
    return h


def test_register_unregister():
    h = _fake_handle()
    _inflight.register(h)
    assert id(h) in _inflight._inflight
    _inflight.unregister(h)
    assert not _inflight._inflight


def test_cancel_all_sweeps_non_terminal():
    running = _fake_handle("t-run")
    done = _fake_handle("t-done", result=object())
    cancelled = _fake_handle("t-cxl", last_status="cancelled")
    for h in (running, done, cancelled):
        _inflight.register(h)

    swept = _inflight.cancel_all()

    assert swept == 1
    running._cancel_remote.assert_called_once()
    done._cancel_remote.assert_not_called()
    cancelled._cancel_remote.assert_not_called()
    assert not _inflight._inflight  # registry emptied either way


def test_cancel_all_idempotent():
    h = _fake_handle()
    _inflight.register(h)
    assert _inflight.cancel_all() == 1
    assert _inflight.cancel_all() == 0
    h._cancel_remote.assert_called_once()


def test_no_hook_outside_ipython():
    with patch.object(_inflight.atexit, "register") as reg:
        # No IPython in sys.modules in a plain test run — but guard anyway.
        with patch.dict(sys.modules):
            sys.modules.pop("IPython", None)
            _inflight.register(_fake_handle())
    reg.assert_not_called()
    assert _inflight._hook_installed is False


def test_hook_installed_inside_ipython_kernel():
    fake_ipython = MagicMock()
    fake_ipython.get_ipython.return_value = object()  # active shell
    with patch.object(_inflight.atexit, "register") as reg:
        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            _inflight.register(_fake_handle())
    reg.assert_called_once_with(_inflight.cancel_all)
    assert _inflight._hook_installed is True


def test_no_hook_when_ipython_imported_but_no_shell():
    fake_ipython = MagicMock()
    fake_ipython.get_ipython.return_value = None  # imported, no kernel
    with patch.object(_inflight.atexit, "register") as reg:
        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            _inflight.register(_fake_handle())
    reg.assert_not_called()
    assert _inflight._hook_installed is False


def test_taskhandle_cancel_remote_unregisters():
    """The real TaskHandle drops out of the registry when cancelled."""
    from krauncher.models import TaskHandle

    client = MagicMock()
    client.broker_url = "http://broker.invalid"
    client.api_key = "k"
    handle = TaskHandle(task_id="t-real", client=client)
    _inflight.register(handle)

    handle._cancel_remote()  # DELETE fails (invalid host) — best-effort

    assert not _inflight._inflight
