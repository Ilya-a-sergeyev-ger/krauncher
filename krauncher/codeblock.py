# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Synthesize a remote task function from a code block.

The application-agnostic base for adapters that submit code *strings* (a
notebook cell, an editor selection) rather than decorated functions. The block
becomes the body of a generated function whose parameters are the input names
and whose return value is the ``{name: value}`` outputs dict. The generated
source is registered in ``linecache`` so ``inspect.getsource`` — which
krauncher's serializer relies on — works on the exec'd function object.

The generated function is exactly what the analyzer classifies and the worker
executes: plain user code, no transport scaffolding. Inputs/outputs therefore
carry JSON-safe values only (enforced in ``krauncher.values``); large/complex
data goes through a data source / volume, not through the function body.
"""

from __future__ import annotations

import ast
import linecache
import textwrap
from itertools import count
from typing import Callable

from .exceptions import SerializationError

_ENTRY = "_kr_cell"
_seq = count(1)

_PROLOGUE = f"""\
def {_ENTRY}({{params}}):
"""

_EPILOGUE = """\
    return {{{returns}}}
"""


def _reject_unsupported(code: str) -> None:
    """Code blocks run inside a function — reject constructs that break there."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SerializationError(f"code block does not parse: {exc}") from exc
    # Only top-level statements matter: `return`/`nonlocal` at module level are
    # already SyntaxErrors, and global/nonlocal inside functions defined by the
    # block are legitimate. A top-level `global` would silently detach the name
    # from the wrapper's locals and break output capture — reject it.
    for node in tree.body:
        if isinstance(node, ast.Global):
            raise SerializationError(
                "top-level `global` is not supported inside a submitted code "
                "block (the block runs as a function)."
            )


def build_code_function(code: str, inputs: list[str], outputs: list[str]) -> Callable:
    """Generate, exec and return the wrapper function for a code block."""
    _reject_unsupported(code)

    params = ", ".join(f"{n}=None" for n in inputs)
    body = textwrap.indent(code.rstrip() + "\n", "    ")
    returns = ", ".join(f"{n!r}: {n}" for n in outputs)

    source = (
        _PROLOGUE.format(params=params)
        + body
        + _EPILOGUE.format(returns=returns)
    )

    # Register in linecache under a unique pseudo-filename so
    # inspect.getsource (used by krauncher's serializer) sees the code.
    filename = f"<krauncher-cell-{next(_seq)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)

    namespace: dict = {}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace[_ENTRY]
