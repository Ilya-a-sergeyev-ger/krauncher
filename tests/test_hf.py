# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for HuggingFace reference detection and sizing (krauncher.hf)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from krauncher.hf import detect_hf_refs, hf_size_mb


def test_detect_load_dataset_literal():
    urls, dynamic = detect_hf_refs('ds = load_dataset("ylecun/mnist")\n')
    assert urls == ["hf://datasets/ylecun/mnist"]
    assert dynamic == []


def test_detect_canonical_dataset_id():
    urls, _ = detect_hf_refs('ds = load_dataset("imdb", split="train")\n')
    assert urls == ["hf://datasets/imdb"]


def test_detect_from_pretrained_model():
    code = 'model = AutoModel.from_pretrained("bert-base-uncased")\n'
    urls, _ = detect_hf_refs(code)
    assert urls == ["hf://models/bert-base-uncased"]


def test_builders_are_not_hub_repos():
    urls, dynamic = detect_hf_refs('ds = load_dataset("json", data_files="x.json")\n')
    assert urls == [] and dynamic == []


def test_local_paths_skipped():
    code = (
        'm = AutoModel.from_pretrained("./checkpoint")\n'
        'ds = load_dataset("/data/mydir")\n'
    )
    urls, dynamic = detect_hf_refs(code)
    assert urls == [] and dynamic == []


def test_dynamic_reference_reported():
    code = "name = pick_model()\nm = AutoModel.from_pretrained(name)\n"
    urls, dynamic = detect_hf_refs(code)
    assert urls == []
    assert dynamic == ["...from_pretrained(...)"]


def test_dedup_and_order():
    code = (
        'a = load_dataset("org/ds1")\n'
        'b = AutoTokenizer.from_pretrained("org/m1")\n'
        'c = load_dataset("org/ds1")\n'
    )
    urls, _ = detect_hf_refs(code)
    assert urls == ["hf://datasets/org/ds1", "hf://models/org/m1"]


def test_syntax_error_is_empty():
    assert detect_hf_refs("def broken(:\n") == ([], [])


def _mock_session(payloads: dict[str, dict]):
    """AsyncClient mock: URL path suffix -> JSON payload."""
    session = AsyncMock()
    session.__aenter__.return_value = session

    async def _get(url, params=None):
        resp = MagicMock()
        for suffix, payload in payloads.items():
            if url.endswith(suffix):
                resp.status_code = 200
                resp.json.return_value = payload
                return resp
        resp.status_code = 404
        return resp

    session.get = _get
    return session


def test_hf_size_mb_sums_siblings():
    payloads = {
        "datasets/ylecun/mnist": {"siblings": [{"size": 10 << 20}, {"size": 1 << 20}]},
    }
    with patch("krauncher.hf.httpx.AsyncClient", return_value=_mock_session(payloads)):
        size = asyncio.run(hf_size_mb(["hf://datasets/ylecun/mnist"]))
    assert size == 11.0


def test_hf_size_mb_none_when_unresolved():
    with patch("krauncher.hf.httpx.AsyncClient", return_value=_mock_session({})):
        assert asyncio.run(hf_size_mb(["hf://datasets/none/gone"])) is None


def test_hf_size_mb_partial_resolution():
    payloads = {"models/org/m1": {"siblings": [{"size": 2 << 20}]}}
    with patch("krauncher.hf.httpx.AsyncClient", return_value=_mock_session(payloads)):
        size = asyncio.run(hf_size_mb(["hf://models/org/m1", "hf://datasets/none/gone"]))
    assert size == 2.0
