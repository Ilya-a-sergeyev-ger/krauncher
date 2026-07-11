# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for krauncher.values — the in/out value transfer primitive."""

import pytest

from krauncher import ValueTransferError
from krauncher.values import decode_outputs, encode_inputs

NS = {"epochs": 3, "name": "bert", "data": [1, 2, 3], "ratio": 0.5}


def test_json_values_pass_raw():
    # JSON-safe values pass through unchanged so the analyzer sees them
    # (epochs/batch_size drive CU estimation) and the task runs on them
    # directly.
    assert encode_inputs(["epochs", "name", "data", "ratio"], NS) == NS


def test_missing_input_rejected():
    with pytest.raises(ValueTransferError, match="not defined"):
        encode_inputs(["nope"], {})


def test_non_json_input_rejected():
    with pytest.raises(ValueTransferError, match="not JSON-serializable"):
        encode_inputs(["f"], {"f": lambda: 1})


def test_per_value_budget_enforced():
    with pytest.raises(ValueTransferError, match="inline budget"):
        encode_inputs(["big"], {"big": "x" * 200}, limit_bytes=100)


def test_total_budget_enforced():
    ns = {"a": "x" * 60, "b": "y" * 60}  # each fits, the sum does not
    with pytest.raises(ValueTransferError, match="total"):
        encode_inputs(["a", "b"], ns, limit_bytes=100)


def test_within_budget_passes():
    ns = {"a": "x" * 30, "b": "y" * 30}
    assert encode_inputs(["a", "b"], ns, limit_bytes=100) == ns


def test_decode_outputs():
    assert decode_outputs({"total": 18, "label": "BERT"}, ["total", "label"]) == {
        "total": 18, "label": "BERT",
    }


def test_decode_missing_output_rejected():
    with pytest.raises(ValueTransferError, match="missing from the task result"):
        decode_outputs({"total": 18}, ["total", "label"])


def test_decode_non_dict_rejected():
    with pytest.raises(ValueTransferError, match="expected the outputs dict"):
        decode_outputs([1, 2, 3], ["x"])


def test_error_is_krauncher_error():
    from krauncher import KrauncherError
    assert issubclass(ValueTransferError, KrauncherError)
