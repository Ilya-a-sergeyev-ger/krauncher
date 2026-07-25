# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for the artifacts round trip — framing, delivery guard, task option."""

import json

import pytest

from krauncher import KrauncherError
from krauncher.models import TaskHandle, TaskResult, _unframe_result


def _frame(output, artifacts):
    """The worker's wire format (cas-worker/src/relay/publisher.py)."""
    if artifacts is None:
        return json.dumps({"output": output}).encode()
    names = sorted(artifacts)
    header = {
        "output": output,
        "artifacts": [{"name": n, "size": len(artifacts[n])} for n in names],
    }
    return json.dumps(header).encode() + b"\n" + b"".join(artifacts[n] for n in names)


# ── wire format ──────────────────────────────────────────────────────


def test_result_without_artifacts_is_unchanged_json():
    assert _unframe_result(_frame({"loss": 0.1}, None)) == {"output": {"loss": 0.1}}


def test_files_survive_the_round_trip():
    files = {"a.png": b"\x89PNG\r\n\x1a\n", "sub/b.txt": b"hello"}
    decoded = _unframe_result(_frame({"steps": 25}, files))
    assert decoded["output"] == {"steps": 25}
    assert decoded["artifacts"] == files


def test_binary_payloads_are_not_corrupted_by_newlines():
    # The body is sliced by declared sizes, not by separators — a file full of
    # newlines must come back byte-identical.
    files = {"log.txt": b"\n\n\n\x00\xff\n"}
    assert _unframe_result(_frame(None, files))["artifacts"] == files


def test_empty_artifact_set_is_distinguishable_from_none():
    assert _unframe_result(_frame({"x": 1}, {}))["artifacts"] == {}
    assert "artifacts" not in _unframe_result(_frame({"x": 1}, None))


def test_zero_length_file_round_trips():
    assert _unframe_result(_frame(None, {"empty": b""}))["artifacts"] == {"empty": b""}


# ── TaskResult ───────────────────────────────────────────────────────


def test_files_lists_names_sorted():
    r = TaskResult(task_id="t", status="completed", artifacts={"b": b"1", "a": b"2"})
    assert r.files == ["a", "b"]


def test_download_writes_relative_paths(tmp_path):
    r = TaskResult(
        task_id="t", status="completed",
        artifacts={"out.png": b"png", "sub/deep.txt": b"text"},
    )
    assert r.download(str(tmp_path)) == 2
    assert (tmp_path / "out.png").read_bytes() == b"png"
    assert (tmp_path / "sub" / "deep.txt").read_bytes() == b"text"


def test_no_artifacts_means_no_files():
    r = TaskResult(task_id="t", status="completed")
    assert r.files == []
    assert r.download() == 0


# ── delivery guard ───────────────────────────────────────────────────


def _handle(declared, status="completed", collected=None):
    h = TaskHandle.__new__(TaskHandle)
    h.task_id = "t-1"
    h._artifacts = declared
    h._result = TaskResult(task_id="t-1", status=status, artifacts=collected)
    return h


def test_declared_but_never_reported_raises():
    # An old worker ignores the declaration; the caller must not be left
    # looking for files that were never collected.
    with pytest.raises(KrauncherError, match="does not support the artifacts API"):
        _handle(True, collected=None)._check_artifacts_delivered()


def test_declared_and_nothing_written_is_fine():
    _handle(True, collected={})._check_artifacts_delivered()


def test_nothing_declared_never_raises():
    _handle(False, collected=None)._check_artifacts_delivered()


def test_failed_task_does_not_raise_about_artifacts():
    # A failed task has a real error to report; do not mask it.
    _handle(True, status="failed", collected=None)._check_artifacts_delivered()


# ── task option ──────────────────────────────────────────────────────


def _render():
    """Top-level so the serializer accepts it."""
    return {}


def test_declaration_reaches_the_task_options():
    from krauncher import KrauncherClient

    client = KrauncherClient(api_key="cas_test", broker_url="http://localhost:1")
    task = client.task(vram_gb=1, artifacts=True)(_render)
    assert task._krauncher_options["artifacts"] is True



# ── name safety ──────────────────────────────────────────────────────


def test_download_refuses_names_that_escape_the_destination(tmp_path):
    # Names come from the worker; a result must not be able to write outside
    # the directory the caller asked for.
    r = TaskResult(
        task_id="t", status="completed",
        artifacts={"../escaped.txt": b"nope"},
    )
    with pytest.raises(KrauncherError, match="escapes the download directory"):
        r.download(str(tmp_path / "out"))
    assert not (tmp_path / "escaped.txt").exists()


def test_download_refuses_absolute_names(tmp_path):
    r = TaskResult(
        task_id="t", status="completed", artifacts={"/tmp/pwned": b"nope"},
    )
    with pytest.raises(KrauncherError, match="escapes the download directory"):
        r.download(str(tmp_path))


def test_download_of_nothing_creates_nothing(tmp_path):
    dest = tmp_path / "unused"
    assert TaskResult(task_id="t", status="completed").download(str(dest)) == 0
    assert not dest.exists()


def test_files_parameter_collides_and_is_rejected():
    from krauncher import KrauncherClient

    client = KrauncherClient(api_key="cas_test", broker_url="http://localhost:1")

    with pytest.raises(KrauncherError, match="collides with the call-time channel"):
        client.task(vram_gb=1)(_task_with_files_param)


def _task_with_files_param(files=None):
    return {}


def test_inconsistent_frame_is_rejected_not_silently_truncated():
    from krauncher import ValueTransferError

    header = json.dumps({"output": None, "artifacts": [{"name": "a", "size": 99}]})
    bad = header.encode() + b"\n" + b"short"

    with pytest.raises(ValueTransferError, match="inconsistent"):
        _unframe_result(bad)
