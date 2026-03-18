# Copyright (c) 2024-2026 Ilya Sergeev. Licensed under the MIT License.

"""Serialize Python functions to code_string for the CaS worker."""

from __future__ import annotations

import ast
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


def _called_names(source: str) -> set[str]:
    """Extract names that appear as function calls in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def _serialize_one(fn: Callable) -> str:
    """Get dedented, decorator-stripped source for a single function."""
    source = inspect.getsource(fn)
    source = _strip_decorators(source)
    return textwrap.dedent(source)


def _collect_helpers(fn: Callable, entry_source: str) -> str:
    """Collect helper functions from the same module that *fn* calls.

    Walks the call graph (breadth-first), serializes each helper, and
    returns them concatenated in dependency order (deepest helpers first).
    """
    module = inspect.getmodule(fn)
    if module is None:
        return ""

    collected: dict[str, str] = {}  # name → source
    visited: set[str] = {fn.__name__}
    queue = list(_called_names(entry_source))

    while queue:
        candidate = queue.pop(0)
        if candidate in visited:
            continue
        visited.add(candidate)

        obj = getattr(module, candidate, None)
        if obj is None or not inspect.isfunction(obj):
            continue
        if getattr(obj, "__module__", None) != fn.__module__:
            continue

        try:
            helper_source = _serialize_one(obj)
        except (OSError, TypeError):
            continue

        collected[candidate] = helper_source
        # Recurse: check what *this* helper calls
        queue.extend(_called_names(helper_source) - visited)

    if not collected:
        return ""

    # Order: helpers in collection order (BFS from entry point).
    # Reverse so deepest dependencies come first.
    parts = list(collected.values())
    parts.reverse()
    return "\n\n".join(parts)


def serialize_function(fn: Callable) -> tuple[str, str]:
    """Serialize a function to (code_string, entry_point).

    The worker injects code_string at module scope and calls
    ``globals()[entry_point](**args)``.  The entry point must be a
    plain, top-level function.  Helper functions defined in the same
    module that are called by the entry point are automatically
    included in the serialized code.

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
        vars_str = ", ".join(fn.__code__.co_freevars)
        kwargs_hint = ", ".join(f"{v}={v}" for v in fn.__code__.co_freevars)
        raise SerializationError(
            f"Function '{name}' captures variables from enclosing scope: {vars_str}. "
            f"Move them into function arguments and pass via kwargs:\n"
            f"  handle = await {name}({kwargs_hint})"
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

    # Collect helper functions from the same module
    helpers = _collect_helpers(fn, code_string)
    if helpers:
        code_string = helpers + "\n\n" + code_string

    try:
        compile(code_string, f"<krauncher:{name}>", "exec")
    except SyntaxError as e:
        raise SerializationError(
            f"Serialized source for '{name}' is not valid Python: {e}"
        ) from e

    return code_string, name
