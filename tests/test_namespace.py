# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for analyze_names + the lenient outputs epilogue (auto namespace)."""

import pytest

from krauncher.codeblock import analyze_names, build_code_source
from krauncher.exceptions import SerializationError


# ---------------------------------------------------------------------------
# analyze_names — inputs (free names)
# ---------------------------------------------------------------------------

def test_simple_free_and_assigned():
    free, assigned = analyze_names("y = x * 2\n")
    assert free == ["x"]
    assert assigned == ["y"]

def test_read_then_write_is_both():
    free, assigned = analyze_names("x = x + 1\n")
    assert free == ["x"]
    assert assigned == ["x"]

def test_write_then_read_not_free():
    free, assigned = analyze_names("x = 5\ny = x\n")
    assert free == []
    assert assigned == ["x", "y"]

def test_augassign_reads_first():
    free, assigned = analyze_names("total += n\n")
    assert free == ["total", "n"]
    assert assigned == ["total"]

def test_for_loop_accumulator():
    free, assigned = analyze_names("s = 0\nfor i in items:\n    s += i\n")
    assert free == ["items"]
    assert assigned == ["s", "i"]

def test_nested_def_free_var():
    code = "def f():\n    return epochs * 2\nr = f()\n"
    free, assigned = analyze_names(code)
    assert free == ["epochs"]
    assert assigned == ["r"]  # f is a code binding, not an output

def test_comprehension_var_does_not_leak():
    free, assigned = analyze_names("sq = [v * v for v in vals]\n")
    assert free == ["vals"]
    assert assigned == ["sq"]

def test_import_shadows_input():
    free, assigned = analyze_names("import torch\nm = torch.zeros(3)\n")
    assert free == []
    assert assigned == ["m"]

def test_builtins_not_free():
    free, _ = analyze_names("y = len(x)\nprint(y)\n")
    assert free == ["x"]

def test_subscript_assign_is_input_not_output():
    free, assigned = analyze_names("arr[0] = 1\n")
    assert free == ["arr"]
    assert assigned == []

def test_attribute_augassign_is_input():
    free, assigned = analyze_names("obj.n += 1\n")
    assert free == ["obj"]
    assert assigned == []

def test_lambda_free_var():
    free, assigned = analyze_names("g = lambda a: a + shift\n")
    assert free == ["shift"]
    assert assigned == ["g"]

def test_class_body_free_var():
    free, assigned = analyze_names("class C:\n    lr = default_lr\n")
    assert free == ["default_lr"]
    assert assigned == []  # class name is a code binding

def test_underscore_names_not_outputs():
    _, assigned = analyze_names("_tmp = 5\nres = _tmp * 2\n")
    assert assigned == ["res"]

def test_with_target_is_output_candidate():
    free, assigned = analyze_names("with open(p) as f:\n    d = f.read()\n")
    assert free == ["p"]
    assert assigned == ["f", "d"]

def test_walrus_binding():
    free, assigned = analyze_names("if (n := count()) > 0:\n    x = n\n")
    assert free == ["count"]
    assert assigned == ["n", "x"]

def test_tuple_unpack():
    free, assigned = analyze_names("a, b = pair\n")
    assert free == ["pair"]
    assert assigned == ["a", "b"]

def test_appearance_order():
    free, assigned = analyze_names("y = b + a\nz = y\n")
    assert free == ["b", "a"]
    assert assigned == ["y", "z"]

def test_decorator_and_default_evaluate_at_def_site():
    code = "def f(x=default_x):\n    return x\n"
    free, _ = analyze_names(code)
    assert free == ["default_x"]

def test_syntax_error_raises():
    with pytest.raises(SerializationError):
        analyze_names("def broken(:\n")


# ---------------------------------------------------------------------------
# lenient outputs epilogue
# ---------------------------------------------------------------------------

def _run_generated(code, inputs, outputs, *, lenient, **kwargs):
    source, entry = build_code_source(code, list(inputs), outputs, lenient_outputs=lenient)
    ns: dict = {}
    exec(source, ns)
    return ns[entry](**kwargs)

def test_lenient_drops_non_json_safe_and_unset():
    code = (
        "y = x * 2\n"
        "model = object()\n"       # non-JSON-safe
        "if False:\n"
        "    never_set = 1\n"      # unset at return
    )
    out = _run_generated(code, ["x"], ["y", "model", "never_set"], lenient=True, x=3)
    assert out == {"y": 6}

def test_lenient_returns_all_when_transferable():
    out = _run_generated("a = 1\nb = 'ok'\n", [], ["a", "b"], lenient=True)
    assert out == {"a": 1, "b": "ok"}

def test_strict_epilogue_unchanged():
    out = _run_generated("y = x + 1\n", ["x"], ["y"], lenient=False, x=1)
    assert out == {"y": 2}

def test_strict_fails_on_unset_name():
    with pytest.raises(NameError):
        _run_generated("pass\n", [], ["missing"], lenient=False)
