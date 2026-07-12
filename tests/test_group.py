# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for TaskGroup envelopes (client.group) and group-aware submission."""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from krauncher import KrauncherClient, KrauncherError, TaskGroup
from krauncher.analyzer import TaskClassification


@pytest.fixture(autouse=True)
def _mock_serializer():
    """Tasks in tests are nested functions — bypass the top-level check."""
    with patch("krauncher.KrauncherClient.serialize_function",
               return_value=("def main(): pass", "main")):
        yield


def _client() -> KrauncherClient:
    return KrauncherClient(api_key="cas_test", broker_url="http://broker.invalid")


def _cls(vram: int) -> TaskClassification:
    return TaskClassification(
        min_vram_gb=vram, tier="light", confidence=1.0, analysis_method="ast",
    )


def test_task_wrapper_stores_options():
    client = _client()

    @client.task(vram_gb=6, gpu_name="L4", data="ds1", disk_gb=15, timeout=120)
    def t(x):
        return x

    opts = t._krauncher_options
    assert opts["vram_gb"] == 6
    assert opts["gpu_name"] == "L4"
    assert opts["data"] == "ds1"
    assert opts["disk_gb"] == 15
    assert opts["timeout"] == 120
    assert t._krauncher_defaults == {}
    assert t._krauncher_cls_cache == [None]


def test_group_vram_floor_from_explicit_pins():
    client = _client()
    client._resolve_sizes = AsyncMock(return_value=None)

    @client.task(vram_gb=6)
    def small(x):
        return x

    @client.task(vram_gb=30)
    def big(x):
        return x

    grp = asyncio.run(client.group(small, big))
    assert grp.vram_floor == math.ceil(30 * 1.1)  # same headroom as single-task pins
    assert grp.group_id.startswith("kr-")


def test_group_classifies_unpinned_members():
    client = _client()
    client._resolve_sizes = AsyncMock(return_value=None)
    client._classify = AsyncMock(return_value=_cls(8))

    @client.task()
    def auto(x):
        return x

    @client.task(vram_gb=2)
    def pinned(x):
        return x

    grp = asyncio.run(client.group(auto, pinned))
    client._classify.assert_called_once()
    assert grp.vram_floor == 8  # classified 8 > ceil(2*1.1)=3


def test_group_conflicting_pins_raise():
    client = _client()

    @client.task(vram_gb=6, gpu_name="L4")
    def a(x):
        return x

    @client.task(vram_gb=6, gpu_name="H100")
    def b(x):
        return x

    with pytest.raises(KrauncherError, match="different gpu_name"):
        asyncio.run(client.group(a, b))


def test_group_disk_envelope_includes_data_sizes():
    client = _client()
    client._resolve_sizes = AsyncMock(return_value=2048.0)  # 2 GB of data

    @client.task(vram_gb=6, data="imagenet", disk_gb=15)
    def a(x):
        return x

    @client.task(vram_gb=6, volume="ckpt", disk_gb=10)
    def b(x):
        return x

    grp = asyncio.run(client.group(a, b))
    client._resolve_sizes.assert_called_once_with(["ckpt", "imagenet"])
    assert grp.disk_gb == 15 + 2  # max member disk + total data


def test_group_requires_decorated_tasks():
    client = _client()
    with pytest.raises(KrauncherError, match="task-decorated"):
        asyncio.run(client.group(lambda: 1))
    with pytest.raises(KrauncherError, match="at least one"):
        asyncio.run(client.group())


def test_submit_applies_group_envelope():
    client = _client()
    client._execute = AsyncMock(return_value=MagicMock(name="handle"))
    grp = TaskGroup(
        group_id="kr-test", client=client, vram_floor=33,
        gpu_name="L40S", provider="runpod", disk_gb=17,
    )

    asyncio.run(client._submit(
        "def main(): pass", "main", {},
        classification=_cls(2), group=grp,
    ))

    (_, _, _, _, cls), opts = client._execute.call_args
    assert cls.min_vram_gb == 33  # raised to the floor
    assert cls.tier == "heavy"
    assert opts["group_id"] == "kr-test"
    assert opts["gpu_name"] == "L40S"
    assert opts["provider"] == "runpod"
    assert opts["disk_gb"] == 17


def test_submit_task_pins_win_over_group():
    client = _client()
    client._execute = AsyncMock(return_value=MagicMock(name="handle"))
    grp = TaskGroup(group_id="kr-test", client=client, vram_floor=3, gpu_name="L4")

    asyncio.run(client._submit(
        "def main(): pass", "main", {},
        classification=_cls(24), group=grp,
        gpu_name="H100", group_id="explicit-group", disk_gb=50,
    ))

    (_, _, _, _, cls), opts = client._execute.call_args
    assert cls.min_vram_gb == 24  # above the floor — untouched
    assert opts["gpu_name"] == "H100"
    assert opts["group_id"] == "explicit-group"
    assert opts["disk_gb"] == 50


def test_group_submit_uses_task_metadata():
    client = _client()
    client._submit = AsyncMock(return_value=MagicMock(name="handle"))

    @client.task(vram_gb=6, pip=["torch"])
    def t(x):
        return x

    grp = TaskGroup(group_id="kr-test", client=client, vram_floor=7)
    asyncio.run(grp.submit(t, x=1))

    (code, entry, kwargs), opts = client._submit.call_args
    assert code == t._krauncher_code and entry == t._krauncher_entry_point
    assert kwargs == {"x": 1}
    assert opts["group"] is grp
    assert opts["vram_gb"] == 6 and opts["pip"] == ["torch"]


def test_group_submit_rejects_plain_function():
    client = _client()
    grp = TaskGroup(group_id="kr-test", client=client)
    with pytest.raises(KrauncherError, match="task-decorated"):
        asyncio.run(grp.submit(lambda: 1))
