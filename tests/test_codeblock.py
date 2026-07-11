# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for krauncher.codeblock and KrauncherClient.run_code."""

import linecache
from unittest.mock import AsyncMock, MagicMock

import pytest

from krauncher import KrauncherClient, SerializationError, ValueTransferError
from krauncher.codeblock import build_code_source
from krauncher.serializer import serialize_function
from krauncher.values import decode_outputs, encode_inputs

NS = {"epochs": 3, "name": "bert", "data": [1, 2, 3], "ratio": 0.5}
CODE = "total = sum(data) * epochs\nlabel = name.upper()\n"
INS = ["epochs", "name", "data", "ratio"]
OUTS = ["total", "label"]


def _exec_entry(source: str, entry: str):
    """Run the synthesized source standalone, as the worker does."""
    ns: dict = {}
    exec(compile(source, "<worker>", "exec"), ns)  # noqa: S102
    return ns[entry]


# ---------------------------------------------------------------------------
# build_code_source
# ---------------------------------------------------------------------------

def test_roundtrip_through_generated_source():
    source, entry = build_code_source(CODE, INS, OUTS)
    assert entry == "_kr_cell"
    kwargs = encode_inputs(INS, NS)
    fn = _exec_entry(source, entry)
    assert decode_outputs(fn(**kwargs), OUTS) == {"total": 18, "label": "BERT"}


def test_source_identical_to_exec_serialize_path():
    """2b invariant: the directly synthesized source must be byte-identical
    to what the pre-2b detour produced (exec via linecache, then
    serialize_function on the function object)."""
    source, entry = build_code_source(CODE, INS, OUTS)

    filename = "<invariance-check>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    ns: dict = {}
    exec(compile(source, filename, "exec"), ns)  # noqa: S102
    code_string, entry2 = serialize_function(ns[entry])

    assert code_string == source
    assert entry2 == entry


def test_generated_source_is_plain_user_code():
    """What the analyzer classifies and the worker runs must be the block
    body, with no transport scaffolding leaking in."""
    source, _ = build_code_source(CODE, INS, OUTS)
    for token in ("pickle", "base64", "_kr_dec", "_kr_enc", "b64"):
        assert token not in source


def test_syntax_error_rejected():
    with pytest.raises(SerializationError, match="does not parse"):
        build_code_source("def broken(:\n", [], [])


def test_toplevel_global_rejected():
    with pytest.raises(SerializationError, match="global"):
        build_code_source("global x\nx = 1", [], ["x"])


def test_nested_global_allowed():
    build_code_source("def f():\n    global q\n    q = 1\nf()", [], [])


# ---------------------------------------------------------------------------
# run_code wiring
# ---------------------------------------------------------------------------

def _make_client() -> KrauncherClient:
    return KrauncherClient(api_key="cas_test", broker_url="http://broker:8000")


@pytest.mark.asyncio
async def test_run_code_wires_through_submit():
    """run_code must synthesize the block source and pass it to _submit with
    the input values as kwargs and options forwarded."""
    client = _make_client()
    client._submit = AsyncMock(return_value=MagicMock(name="handle"))

    inputs = {k: NS[k] for k in INS}
    await client.run_code(CODE, inputs=inputs, outputs=OUTS, pip=["torch"], timeout=120)

    (code_string, entry, kwargs), options = client._submit.call_args
    assert kwargs == inputs
    assert options == {"pip": ["torch"], "timeout": 120}
    # The submitted source is the generated block function.
    fn = _exec_entry(code_string, entry)
    assert decode_outputs(fn(**kwargs), OUTS) == {"total": 18, "label": "BERT"}


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
