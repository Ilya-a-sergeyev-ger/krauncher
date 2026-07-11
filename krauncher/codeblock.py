# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Synthesize remote-task source from a code block.

The application-agnostic base for adapters that submit code *strings* (a
notebook cell, an editor selection) rather than decorated functions. The block
becomes the body of a generated function whose parameters are the input names
and whose return value is the ``{name: value}`` outputs dict. The synthesized
source is submitted directly — the same ``(code_string, entry_point)`` shape
``serialize_function`` produces for decorated functions.

The generated function is exactly what the analyzer classifies and the worker
executes: plain user code, no transport scaffolding. Inputs/outputs therefore
carry JSON-safe values only (enforced in ``krauncher.values``); large/complex
data goes through a data source / volume, not through the function body.
"""

from __future__ import annotations

import ast
import textwrap

from .exceptions import SerializationError

_ENTRY = "_kr_cell"

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


def build_code_source(code: str, inputs: list[str], outputs: list[str]) -> tuple[str, str]:
    """Synthesize ``(code_string, entry_point)`` for a code block."""
    _reject_unsupported(code)

    params = ", ".join(f"{n}=None" for n in inputs)
    body = textwrap.indent(code.rstrip() + "\n", "    ")
    returns = ", ".join(f"{n!r}: {n}" for n in outputs)

    source = (
        _PROLOGUE.format(params=params)
        + body
        + _EPILOGUE.format(returns=returns)
    )

    # The body parses on its own, but the assembled wrapper can still be
    # invalid (e.g. a block indented with tabs) — mirror serialize_function's
    # compile guard so the failure happens client-side with a clear message.
    try:
        compile(source, f"<krauncher:{_ENTRY}>", "exec")
    except SyntaxError as exc:
        raise SerializationError(
            f"synthesized source for the code block is not valid Python: {exc}"
        ) from exc

    return source, _ENTRY
