# Copyright (c) 2024-2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for krauncher.serializer."""

import pytest

from krauncher.exceptions import SerializationError
from krauncher.serializer import serialize_function


# -- Valid cases ---------------------------------------------------------------


def simple_function():
    return 42


def function_with_imports():
    import math
    return math.pi


def function_with_args(x, y=10):
    return x + y


class TestSerializeValid:
    def test_simple_function(self):
        code, entry = serialize_function(simple_function)
        assert entry == "simple_function"
        assert "def simple_function" in code
        assert "return 42" in code

    def test_function_with_imports(self):
        code, entry = serialize_function(function_with_imports)
        assert entry == "function_with_imports"
        assert "import math" in code

    def test_function_with_args(self):
        code, entry = serialize_function(function_with_args)
        assert entry == "function_with_args"
        assert "x, y=10" in code or "x, y = 10" in code

    def test_code_compiles(self):
        code, _ = serialize_function(simple_function)
        compile(code, "<test>", "exec")

    def test_code_is_dedented(self):
        code, _ = serialize_function(simple_function)
        # First line should start with "def", no leading whitespace
        assert code.startswith("def ")

    def test_decorators_stripped(self):
        """Decorator lines above def should be removed."""
        # We can't easily add a real decorator in a test without side effects,
        # so test the internal _strip_decorators function directly.
        from krauncher.serializer import _strip_decorators

        source = '@client.task(vram_gb=8)\ndef my_func(x):\n    return x\n'
        result = _strip_decorators(source)
        assert result == 'def my_func(x):\n    return x\n'

    def test_multi_decorator_stripped(self):
        from krauncher.serializer import _strip_decorators

        source = '@decorator1\n@decorator2(foo=1)\ndef func():\n    pass\n'
        result = _strip_decorators(source)
        assert result == 'def func():\n    pass\n'


# -- Invalid cases -------------------------------------------------------------


class TestSerializeInvalid:
    def test_reject_lambda(self):
        fn = lambda x: x  # noqa: E731
        with pytest.raises(SerializationError, match="Lambda"):
            serialize_function(fn)

    def test_reject_builtin(self):
        with pytest.raises(SerializationError, match="plain function"):
            serialize_function(len)

    def test_reject_bound_method(self):
        class Foo:
            def bar(self):
                pass

        with pytest.raises(SerializationError, match="plain function"):
            serialize_function(Foo().bar)

    def test_reject_closure_with_kwargs_hint(self):
        x = 10

        def closure_fn():
            return x

        with pytest.raises(SerializationError, match="pass via kwargs"):
            serialize_function(closure_fn)

    def test_closure_error_lists_captured_vars(self):
        a, b = 1, 2

        def captures_two():
            return a + b

        with pytest.raises(SerializationError, match="a, b"):
            serialize_function(captures_two)

    def test_reject_nested_function(self):
        def outer():
            def inner():
                return 1
            return inner

        with pytest.raises(SerializationError, match="nested"):
            serialize_function(outer())

    def test_reject_not_callable(self):
        with pytest.raises(SerializationError, match="callable"):
            serialize_function(42)


# -- Multi-function serialization ---------------------------------------------


class TestMultiFunctionSerialization:
    def test_helpers_included(self):
        """Helper functions called by entry point are included."""
        from tests._helpers_fixture import train

        code, entry = serialize_function(train)
        assert entry == "train"
        assert "def preprocess" in code
        assert "def normalize" in code
        assert "def augment" in code

    def test_helpers_before_entry_point(self):
        """Helpers appear before the entry point in code_string."""
        from tests._helpers_fixture import train

        code, _ = serialize_function(train)
        # Entry point should be last
        last_def = code.rfind("\ndef train(")
        assert last_def > 0, "entry point should not be first"
        # All helpers should be above it
        assert code.find("def normalize") < last_def
        assert code.find("def augment") < last_def
        assert code.find("def preprocess") < last_def

    def test_no_helpers_for_standalone(self):
        """Function with no helper calls produces single-function code."""
        from tests._helpers_fixture import standalone

        code, entry = serialize_function(standalone)
        assert entry == "standalone"
        assert code.count("def ") == 1

    def test_combined_code_compiles(self):
        """Combined code_string with helpers is valid Python."""
        from tests._helpers_fixture import train

        code, _ = serialize_function(train)
        compile(code, "<test>", "exec")

    def test_combined_code_executes(self):
        """Combined code_string actually runs correctly."""
        from tests._helpers_fixture import train

        code, entry = serialize_function(train)
        ns: dict = {}
        exec(code, ns)
        result = ns[entry](data=[1, 2, 3, 4, 5], epochs=3)
        assert result["epochs"] == 3
        assert isinstance(result["result"], (int, float))
