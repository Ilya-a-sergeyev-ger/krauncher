# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""KRAUNCHER_VRAM_HEADROOM — the safety factor on every VRAM requirement."""

import math

import pytest

from krauncher.analyzer import (
    _VRAM_HEADROOM_DEFAULT,
    _VRAM_HEADROOM_ENV,
    classify_explicit,
    vram_headroom,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_VRAM_HEADROOM_ENV, raising=False)


def test_default_is_five_percent():
    assert vram_headroom() == _VRAM_HEADROOM_DEFAULT == 1.05


def test_env_overrides(monkeypatch):
    monkeypatch.setenv(_VRAM_HEADROOM_ENV, "1.25")
    assert vram_headroom() == 1.25


def test_read_per_call_not_at_import(monkeypatch):
    """A value set after import still applies — notebooks set it mid-session."""
    assert vram_headroom() == 1.05
    monkeypatch.setenv(_VRAM_HEADROOM_ENV, "1.4")
    assert vram_headroom() == 1.4


@pytest.mark.parametrize("bad", ["", "lots", "0.8", "-1"])
def test_unusable_values_fall_back(monkeypatch, bad):
    """Below 1.0 would ask for less VRAM than the estimate; not a headroom."""
    monkeypatch.setenv(_VRAM_HEADROOM_ENV, bad)
    assert vram_headroom() == _VRAM_HEADROOM_DEFAULT


def test_explicit_pin_uses_it(monkeypatch):
    assert classify_explicit(20).min_vram_gb == math.ceil(20 * 1.05)
    monkeypatch.setenv(_VRAM_HEADROOM_ENV, "1.5")
    assert classify_explicit(20).min_vram_gb == 30


def test_analyzed_estimate_lands_on_a_23gb_card():
    """The case this default was chosen for: a 21 GB requirement must not be
    inflated past a 23 GB card."""
    from krauncher.analyzer import AnalyzerClient
    c = AnalyzerClient._parse_result({"min_hardware": {"min_vram_gb": 21}})
    assert c.min_vram_gb == 23


# --- KRAUNCHER_ prefix, with the original CAS_ spelling still accepted -----


def test_cas_spelling_still_read(monkeypatch):
    monkeypatch.setenv("CAS_VRAM_HEADROOM", "1.3")
    assert vram_headroom() == 1.3


def test_krauncher_wins_over_cas(monkeypatch):
    monkeypatch.setenv("CAS_VRAM_HEADROOM", "1.3")
    monkeypatch.setenv("KRAUNCHER_VRAM_HEADROOM", "1.2")
    assert vram_headroom() == 1.2


@pytest.fixture(autouse=True)
def _clean_cas_env(monkeypatch):
    monkeypatch.delenv("CAS_VRAM_HEADROOM", raising=False)


def test_settings_read_both_prefixes(monkeypatch):
    from krauncher._env import setting
    monkeypatch.delenv("KRAUNCHER_API_KEY", raising=False)
    monkeypatch.setenv("CAS_API_KEY", "old-spelling")
    assert setting("API_KEY") == "old-spelling"
    monkeypatch.setenv("KRAUNCHER_API_KEY", "new-spelling")
    assert setting("API_KEY") == "new-spelling"
    assert setting("NOTHING_SET", "fallback") == "fallback"
