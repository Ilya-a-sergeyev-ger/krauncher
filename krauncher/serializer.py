"""Serialize Python functions to code_string for the CaS worker."""

from __future__ import annotations

import inspect
import textwrap
from typing import Callable

from .exceptions import SerializationError


def _strip_decorators(source: str) -> str:
    """Remove decorator lines from function source.

    ``inspect.getsource()`` includes ``@decorator`` lines above ``def``.
    The worker only needs the bare ``def ...`` block.
    Handles multi-line decorators (parenthesized arguments).
    """
    lines = source.splitlines(keepends=True)
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].lstrip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            break
        idx += 1
    return "".join(lines[idx:])


def serialize_function(fn: Callable) -> tuple[str, str]:
    """Serialize a function to (code_string, entry_point).

    The worker injects code_string at module scope and calls
    ``globals()[entry_point](**args)``. Therefore the function must be
    a plain, top-level, self-contained definition with all imports
    inside the body.

    Returns:
        Tuple of (code_string, entry_point_name).

    Raises:
        SerializationError: If the function cannot be serialized.
    """
    if not callable(fn):
        raise SerializationError(f"Expected a callable, got {type(fn).__name__}")

    if not inspect.isfunction(fn):
        raise SerializationError(
            f"Expected a plain function, got {type(fn).__name__}. "
            "Lambdas, bound methods, and built-in functions are not supported."
        )

    name = fn.__name__
    if name == "<lambda>":
        raise SerializationError("Lambda functions cannot be serialized. Use a named function.")

    if fn.__code__.co_freevars:
        raise SerializationError(
            f"Function '{name}' captures variables from enclosing scope: "
            f"{fn.__code__.co_freevars}. "
            "CaS tasks must be self-contained with no closures."
        )

    if "<locals>" in fn.__qualname__:
        raise SerializationError(
            f"Function '{name}' is nested inside another function "
            f"(qualname='{fn.__qualname__}'). "
            "CaS tasks must be top-level functions."
        )

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as e:
        raise SerializationError(
            f"Cannot retrieve source code for '{name}': {e}. "
            "The function must be defined in a .py file (not interactive shell or exec)."
        ) from e

    # Strip decorator lines — inspect.getsource() includes them,
    # but the worker only needs the bare function definition.
    source = _strip_decorators(source)

    code_string = textwrap.dedent(source)

    try:
        compile(code_string, f"<krauncher:{name}>", "exec")
    except SyntaxError as e:
        raise SerializationError(
            f"Serialized source for '{name}' is not valid Python: {e}"
        ) from e

    return code_string, name
