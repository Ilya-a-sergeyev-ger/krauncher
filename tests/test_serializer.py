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

    def test_reject_closure(self):
        x = 10

        def closure_fn():
            return x

        with pytest.raises(SerializationError, match="closures"):
            serialize_function(closure_fn)

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
