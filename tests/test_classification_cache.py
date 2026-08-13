# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""The per-task classification cache: one analysis per distinct call.

A decorated task is analysed once and the answer reused — but the call
arguments are part of what was analysed (epochs, batch size and the like reach
the analyzer as kwargs and set the iteration count), and the result travels to
the broker with every submission, where it picks the GPU and sizes the hold.
So a repeat call must be free and a different call must be re-analysed.
"""

import pytest

from krauncher import KrauncherClient
from krauncher.analyzer import TaskClassification
from krauncher.KrauncherClient import _kwargs_cache_key

CODE = "def train(epochs=1, batch_size=8):\n    return 1\n"
DEFAULTS = {"epochs": 1, "batch_size": 8}


@pytest.fixture
def calls(monkeypatch):
    """A client whose analyzer records the kwargs of every classification."""
    seen: list[dict] = []

    class _Analyzer:
        async def classify(self, code, dataset_mb=None, kwargs=None):
            seen.append(dict(kwargs or {}))
            return TaskClassification(
                min_vram_gb=8, tier="light", confidence=1.0,
                analysis_method="ast", compute_units=1000.0 * len(seen),
            )

    monkeypatch.setattr(KrauncherClient, "_analyzer", property(lambda self: _Analyzer()))
    return seen


@pytest.fixture
def client():
    return KrauncherClient(api_key="cas_test", estimate_only=True)


async def _submit(client, cache, **kwargs):
    handle = await client._submit(
        CODE, "train", kwargs, func_defaults=DEFAULTS, classification_cache=cache,
    )
    return handle.classification


async def test_repeat_call_reuses_the_analysis(client, calls):
    cache: dict = {}

    await _submit(client, cache, epochs=3)
    await _submit(client, cache, epochs=3)

    assert calls == [{"epochs": 3, "batch_size": 8}]


async def test_changed_argument_is_re_analysed(client, calls):
    cache: dict = {}

    first = await _submit(client, cache, epochs=1)
    second = await _submit(client, cache, epochs=100, batch_size=64)

    assert calls == [
        {"epochs": 1, "batch_size": 8},
        {"epochs": 100, "batch_size": 64},
    ]
    # The second submission must not carry the first call's forecast.
    assert first.compute_units != second.compute_units


async def test_a_sweep_analyses_each_point_once(client, calls):
    cache: dict = {}

    for batch_size in (8, 16, 32, 8, 16):
        await _submit(client, cache, batch_size=batch_size)

    assert [c["batch_size"] for c in calls] == [8, 16, 32]


async def test_defaults_and_explicit_values_are_the_same_call(client, calls):
    """Passing a default explicitly is the same analysis, not a second one."""
    cache: dict = {}

    await _submit(client, cache, epochs=1)
    await _submit(client, cache)                    # epochs=1 via the default

    assert len(calls) == 1


def test_key_ignores_values_the_analyzer_never_sees():
    """Only scalars are sent, so only scalars may cost a re-analysis."""
    base = {"epochs": 3, "rows": [1, 2, 3]}
    changed = {"epochs": 3, "rows": [4, 5]}

    assert _kwargs_cache_key(base) == _kwargs_cache_key(changed)
    assert _kwargs_cache_key(base) != _kwargs_cache_key({"epochs": 4})


def test_key_is_order_independent():
    assert _kwargs_cache_key({"a": 1, "b": 2}) == _kwargs_cache_key({"b": 2, "a": 1})
