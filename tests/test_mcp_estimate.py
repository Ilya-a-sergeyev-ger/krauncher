# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""The MCP `estimate` path: one request shape whether or not a key is set.

The key selects WHICH analyzer answers, never WHAT is sent — a keyed and a
keyless call on the same code must hand the analyzer the same source. These
tests pin that, plus the fallbacks around it (unusable key, quota, bad code).
"""

import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parents[1] / "mcp"
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

server = pytest.importorskip(
    "krauncher_mcp.server", reason="krauncher-mcp is not installed",
)

from krauncher.analyzer import AnalyzerClient, TaskClassification  # noqa: E402
from krauncher.exceptions import KrauncherError  # noqa: E402

CODE = """def train(steps=200):
    import torch
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").cuda()
    for _ in range(steps):
        model(input_ids=torch.zeros(8, 128, dtype=torch.long).cuda())
    return 1
"""


@pytest.fixture(autouse=True)
def _reset_module_clients(monkeypatch):
    """The server caches both clients in module globals."""
    monkeypatch.setattr(server, "_client", None)
    monkeypatch.setattr(server, "_keyless", None)


class _Recorder:
    """Stands in for an AnalyzerClient: records the source it was asked about."""

    def __init__(self, tag):
        self.tag = tag
        self.seen = []
        self.seen_kwargs = []

    async def classify(self, code, dataset_mb=None, kwargs=None):
        self.seen.append(code)
        self.seen_kwargs.append(kwargs)
        return TaskClassification(
            min_vram_gb=8, tier="light", confidence=1.0, analysis_method="ast",
            compute_units=7500.0, cu_io=1000.0, cu_setup=3000.0,
            extra_debug={"detected_iterations": 4690,
                         "iteration_basis": "literal_loop"},
        )


class _FakeKeyedClient:
    """KrauncherClient stub: hands back a recorder, or refuses to resolve one."""

    def __init__(self, recorder=None, raises=None):
        self._recorder = recorder
        self._raises = raises
        self.tags = None

    def analyzer(self, *, source="api", surface=None):
        if self._raises is not None:
            raise self._raises
        self.tags = (source, surface)
        return self._recorder


# ---------------------------------------------------------------------------
# The unified path
# ---------------------------------------------------------------------------

async def test_keyed_and_keyless_send_the_same_source(monkeypatch):
    keyed, keyless = _Recorder("keyed"), _Recorder("keyless")

    monkeypatch.setattr(server, "_get_client", lambda: _FakeKeyedClient(keyed))
    await server._classify(CODE)

    monkeypatch.setattr(server, "_get_client", lambda: None)
    monkeypatch.setattr(server, "_get_keyless", lambda: keyless)
    await server._classify(CODE)

    assert keyed.seen == keyless.seen == [CODE]


async def test_keyed_source_is_not_wrapped(monkeypatch):
    """The submission wrapper (`_kr_cell`) belongs to run_code, not here."""
    keyed = _Recorder("keyed")
    monkeypatch.setattr(server, "_get_client", lambda: _FakeKeyedClient(keyed))

    await server._classify(CODE)

    assert "_kr_cell" not in keyed.seen[0]


def test_keyed_branch_is_tagged_as_mcp(monkeypatch):
    client = _FakeKeyedClient(_Recorder("keyed"))
    monkeypatch.setattr(server, "_get_client", lambda: client)

    server._analyzer()

    assert client.tags == ("mcp", "mcp")


def test_keyless_branch_is_tagged_as_mcp(monkeypatch):
    monkeypatch.setattr(server, "_get_client", lambda: None)
    monkeypatch.setattr(server, "_analyzer_url", lambda: "https://analyzer.test")

    ac = server._analyzer()

    assert isinstance(ac, AnalyzerClient)
    assert ac._source == "mcp"
    assert ac._headers["X-Client-Surface"] == "mcp"
    assert "X-Analyzer-Token" not in ac._headers


def test_both_branches_share_one_timeout(monkeypatch):
    """A key must not shorten the analysis budget (keyed default is 10s)."""
    monkeypatch.setattr(server, "_get_client", lambda: None)
    monkeypatch.setattr(server, "_analyzer_url", lambda: "https://analyzer.test")

    assert server._analyzer()._timeout == server._ANALYZER_TIMEOUT


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def test_unusable_key_degrades_to_keyless(monkeypatch):
    """A revoked key or an unreachable broker must not kill the estimate."""
    client = _FakeKeyedClient(raises=KrauncherError("cannot reach broker"))
    keyless = _Recorder("keyless")
    monkeypatch.setattr(server, "_get_client", lambda: client)
    monkeypatch.setattr(server, "_get_keyless", lambda: keyless)

    assert server._analyzer() is keyless


async def test_quota_error_surfaces_the_registration_cta(monkeypatch):
    class _Resp:
        status_code = 429

        @staticmethod
        def json():
            return {"detail": "Anonymous daily limit reached. Register free at ..."}

    class _QuotaError(Exception):
        response = _Resp()

    async def _raise(_code, _run_args=None):
        raise _QuotaError()

    monkeypatch.setattr(server, "_classify", _raise)
    out = await server.estimate_gpu_time_and_cost(CODE)

    assert out["error"].startswith("Anonymous daily limit reached")
    assert out["confidence"] == 0.0


async def test_unparseable_code_is_reported_before_any_request(monkeypatch):
    async def _fail(_code, _run_args=None):
        raise AssertionError("must not reach the analyzer")

    monkeypatch.setattr(server, "_classify", _fail)
    out = await server.estimate_gpu_time_and_cost("def broken(:\n    pass\n")

    assert out["error"].startswith("code does not parse")
    assert out["compute_sec"] is None


async def test_analyzer_failure_returns_the_contract_shape(monkeypatch):
    async def _raise(_code, _run_args=None):
        raise RuntimeError("analyzer down")

    monkeypatch.setattr(server, "_classify", _raise)
    out = await server.estimate_gpu_time_and_cost(CODE)

    assert out["error"].startswith("estimate unavailable: RuntimeError")
    assert out["reference_card"] == server.REFERENCE_CARD
    assert out["knobs"] == [] and out["findings"] == []


async def test_successful_estimate_reports_reference_card_seconds(monkeypatch):
    recorder = _Recorder("keyed")
    monkeypatch.setattr(server, "_get_client", lambda: _FakeKeyedClient(recorder))

    out = await server.estimate_gpu_time_and_cost(CODE)

    # CU/1000 == seconds on the reference card; compute = total - io - setup.
    assert (out["compute_sec"], out["setup_sec"], out["io_sec"]) == (3.5, 3.0, 1.0)
    assert out["min_vram_gb"] == 8
    assert "error" not in out


async def test_iteration_basis_reaches_the_agent(monkeypatch):
    """The estimate scales with the step count, so where that count came from
    travels with it."""
    recorder = _Recorder("keyed")
    monkeypatch.setattr(server, "_get_client", lambda: _FakeKeyedClient(recorder))

    out = await server.estimate_gpu_time_and_cost(CODE)

    assert out["iterations"] == 4690
    assert out["iteration_basis"] == "literal_loop"


async def test_run_args_reach_the_analyzer(monkeypatch):
    """The values the job will be called with are what makes a parameterised
    schedule resolvable."""
    recorder = _Recorder("keyed")
    monkeypatch.setattr(server, "_get_client", lambda: _FakeKeyedClient(recorder))

    await server.estimate_gpu_time_and_cost(CODE, {"steps": 200, "batch_size": 32})

    assert recorder.seen_kwargs == [{"steps": 200, "batch_size": 32}]


async def test_no_run_args_sends_none_not_an_empty_dict(monkeypatch):
    """An empty mapping would read as 'the call takes no arguments'."""
    recorder = _Recorder("keyed")
    monkeypatch.setattr(server, "_get_client", lambda: _FakeKeyedClient(recorder))

    await server.estimate_gpu_time_and_cost(CODE)
    await server.estimate_gpu_time_and_cost(CODE, {})

    assert recorder.seen_kwargs == [None, None]


async def test_iteration_fields_are_null_on_an_older_analyzer(monkeypatch):
    """The analyzer may not report them; the contract keeps its shape."""

    class _Old:
        async def classify(self, code, dataset_mb=None, kwargs=None):
            return TaskClassification(
                min_vram_gb=8, tier="light", confidence=1.0,
                analysis_method="ast", compute_units=7500.0,
            )

    monkeypatch.setattr(server, "_get_client", lambda: _FakeKeyedClient(_Old()))

    out = await server.estimate_gpu_time_and_cost(CODE)

    assert out["iterations"] is None and out["iteration_basis"] is None
