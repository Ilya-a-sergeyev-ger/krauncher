# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for krauncher.codeblock and KrauncherClient.run_code."""

import inspect
from unittest.mock import MagicMock

import pytest

from krauncher import KrauncherClient, SerializationError, ValueTransferError
from krauncher.codeblock import build_code_function
from krauncher.serializer import serialize_function
from krauncher.values import decode_outputs, encode_inputs

NS = {"epochs": 3, "name": "bert", "data": [1, 2, 3], "ratio": 0.5}
CODE = "total = sum(data) * epochs\nlabel = name.upper()\n"
INS = ["epochs", "name", "data", "ratio"]
OUTS = ["total", "label"]


# ---------------------------------------------------------------------------
# build_code_function
# ---------------------------------------------------------------------------

def test_roundtrip_through_generated_function():
    kwargs = encode_inputs(INS, NS)
    fn = build_code_function(CODE, INS, OUTS)
    assert decode_outputs(fn(**kwargs), OUTS) == {"total": 18, "label": "BERT"}


def test_serializer_worker_simulation():
    """krauncher's serialize_function must see the generated source, and the
    serialized string must execute standalone (as the worker does)."""
    fn = build_code_function(CODE, INS, OUTS)
    code_string, entry = serialize_function(fn)
    assert entry == "_kr_cell"
    ns: dict = {}
    exec(compile(code_string, "<worker>", "exec"), ns)  # noqa: S102
    kwargs = encode_inputs(INS, NS)
    assert decode_outputs(ns[entry](**kwargs), OUTS) == {"total": 18, "label": "BERT"}


def test_generated_source_is_plain_user_code():
    """What the analyzer classifies and the worker runs must be the block
    body, with no transport scaffolding leaking in."""
    src = inspect.getsource(build_code_function(CODE, INS, OUTS))
    for token in ("pickle", "base64", "_kr_dec", "_kr_enc", "b64"):
        assert token not in src


def test_syntax_error_rejected():
    with pytest.raises(SerializationError, match="does not parse"):
        build_code_function("def broken(:\n", [], [])


def test_toplevel_global_rejected():
    with pytest.raises(SerializationError, match="global"):
        build_code_function("global x\nx = 1", [], ["x"])


def test_nested_global_allowed():
    build_code_function("def f():\n    global q\n    q = 1\nf()", [], [])


# ---------------------------------------------------------------------------
# run_code wiring
# ---------------------------------------------------------------------------

def _make_client() -> KrauncherClient:
    return KrauncherClient(api_key="cas_test", broker_url="http://broker:8000")


@pytest.mark.asyncio
async def test_run_code_wires_through_task():
    """run_code must build the block function and submit it through task()
    with the input values as kwargs and options forwarded."""
    client = _make_client()
    captured: dict = {}

    def fake_task(**options):
        captured["options"] = options

        def decorator(fn):
            captured["fn"] = fn

            async def submit(**kwargs):
                captured["kwargs"] = kwargs
                return MagicMock(name="handle")

            return submit
        return decorator

    client.task = fake_task
    await client.run_code(
        CODE, inputs={k: NS[k] for k in INS}, outputs=OUTS,
        pip=["torch"], timeout=120,
    )

    assert captured["options"] == {"pip": ["torch"], "timeout": 120}
    assert captured["kwargs"] == {k: NS[k] for k in INS}
    # The captured fn is the generated block function itself.
    assert decode_outputs(captured["fn"](**captured["kwargs"]), OUTS) == {
        "total": 18, "label": "BERT",
    }


@pytest.mark.asyncio
async def test_run_code_budget_covers_code_plus_inputs():
    """Inputs are checked against the budget remaining after the code."""
    client = _make_client()
    huge = "x" * (16 * 1024 * 1024)  # fills the whole budget as a value
    with pytest.raises(ValueTransferError, match="inline budget"):
        await client.run_code("y = x\n", inputs={"x": huge}, outputs=["y"])


@pytest.mark.asyncio
async def test_run_code_oversized_code_rejected():
    client = _make_client()
    huge_code = "# pad\n" * (3 * 1024 * 1024)  # > 16 MB of source
    with pytest.raises(ValueTransferError, match="code block alone"):
        await client.run_code(huge_code)
