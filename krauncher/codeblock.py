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
import builtins as _builtins
import textwrap

from .exceptions import SerializationError

_ENTRY = "_kr_cell"

_PROLOGUE = f"""\
def {_ENTRY}({{params}}):
"""

_EPILOGUE = """\
    return {{{returns}}}
"""

# Lenient epilogue for auto-detected outputs: names may be unset (conditional
# assignment) or non-JSON-safe (a model, a tensor) — return what transfers,
# silently drop the rest; the adapter reports the missing names.
_EPILOGUE_LENIENT = """\
    import json as _kr_json
    _kr_out = {{}}
    _kr_locals = locals()
    for _kr_name in {names!r}:
        if _kr_name in _kr_locals:
            try:
                _kr_json.dumps(_kr_locals[_kr_name])
            except (TypeError, ValueError):
                continue
            _kr_out[_kr_name] = _kr_locals[_kr_name]
    return _kr_out
"""

_BUILTIN_NAMES = frozenset(dir(_builtins))


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


def _scope_free(node: ast.AST) -> set[str]:
    """Free names of a nested scope (function / lambda / class / comprehension)
    — loads not bound inside the scope. Over-reports rather than misses:
    callers filter candidates against the actual namespace anyway.
    """
    loads: set[str] = set()
    bounds: set[str] = set()
    children: list[ast.AST] = []

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg):
            if arg is not None:
                bounds.add(arg.arg)
        children = node.body if isinstance(node.body, list) else [node.body]
    elif isinstance(node, ast.ClassDef):
        children = list(node.body)
    elif isinstance(node, ast.DictComp):
        children = [node.key, node.value]
    else:  # ListComp / SetComp / GeneratorExp
        children = [node.elt]
    for gen in getattr(node, "generators", []):
        for n in ast.walk(gen.target):
            if isinstance(n, ast.Name):
                bounds.add(n.id)
        children.append(gen.iter)
        children.extend(gen.ifs)

    class _V(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name) -> None:
            (loads if isinstance(n.ctx, ast.Load) else bounds).add(n.id)

        def visit_Global(self, n: ast.Global) -> None:
            loads.update(n.names)

        def visit_Nonlocal(self, n: ast.Nonlocal) -> None:
            loads.update(n.names)

        def _nested(self, n: ast.AST) -> None:
            # Decorators / defaults / bases evaluate in *this* scope.
            for d in getattr(n, "decorator_list", []):
                self.visit(d)
            if hasattr(n, "args") and not isinstance(n, ast.ClassDef):
                for d in (*n.args.defaults, *(x for x in n.args.kw_defaults if x)):
                    self.visit(d)
            for b in getattr(n, "bases", []):
                self.visit(b)
            for kw in getattr(n, "keywords", []):
                self.visit(kw.value)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bounds.add(n.name)
            loads.update(_scope_free(n))

        visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _nested
        visit_Lambda = _nested
        visit_ListComp = visit_SetComp = visit_DictComp = _nested
        visit_GeneratorExp = _nested

        def visit_Import(self, n: ast.Import) -> None:
            for alias in n.names:
                bounds.add(alias.asname or alias.name.split(".")[0])

        visit_ImportFrom = visit_Import

    v = _V()
    for child in children:
        v.visit(child)
    return loads - bounds


class _NameScan(ast.NodeVisitor):
    """Linear document-order scan of a code block's top level.

    Records the first load and first binding position per name; nested scopes
    contribute their free names as loads at the definition site. Bindings are
    split into value bindings (assignment-like — output candidates) and code
    bindings (imports, def/class — shadow inputs but are not values).
    """

    def __init__(self) -> None:
        self._pos = 0
        self.first_load: dict[str, int] = {}
        self.first_store: dict[str, int] = {}
        self.value_store: dict[str, int] = {}

    def _tick(self) -> int:
        self._pos += 1
        return self._pos

    def _load(self, name: str) -> None:
        self.first_load.setdefault(name, self._tick())

    def _bind(self, name: str, value: bool = True) -> None:
        pos = self._tick()
        self.first_store.setdefault(name, pos)
        if value:
            self.value_store.setdefault(name, pos)

    def _loads_from_scope(self, node: ast.AST) -> None:
        for name in sorted(_scope_free(node)):
            self._load(name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._load(node.id)
        elif isinstance(node.ctx, ast.Store):
            self._bind(node.id)
        else:  # Del unbinds — shadows inputs, not a value
            self._bind(node.id, value=False)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)  # RHS evaluates first: `x = x + 1` loads x
        for t in node.targets:
            self.visit(t)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.annotation)
        if isinstance(node.target, ast.Name):
            if node.value is not None:  # bare `x: int` does not bind
                self._bind(node.target.id)
        else:
            self.visit(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._load(node.target.id)  # `x += y` reads x before storing
            self.visit(node.value)
            self._bind(node.target.id)
        else:
            self.visit(node.target)  # x[0] += 1: Name x has Load ctx
            self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind(node.target.id)

    def _visit_for(self, node: ast.For) -> None:
        self.visit(node.iter)  # iterable evaluates before target binds
        self.visit(node.target)
        for stmt in (*node.body, *node.orelse):
            self.visit(stmt)

    visit_For = visit_AsyncFor = _visit_for

    def _visit_def(self, node: ast.AST) -> None:
        for d in getattr(node, "decorator_list", []):
            self.visit(d)
        if hasattr(node, "args") and not isinstance(node, ast.ClassDef):
            for d in (*node.args.defaults, *(x for x in node.args.kw_defaults if x)):
                self.visit(d)
        for b in getattr(node, "bases", []):
            self.visit(b)
        for kw in getattr(node, "keywords", []):
            self.visit(kw.value)
        self._bind(node.name, value=False)
        self._loads_from_scope(node)

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _visit_def

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for d in (*node.args.defaults, *(x for x in node.args.kw_defaults if x)):
            self.visit(d)
        self._loads_from_scope(node)

    def _visit_comp(self, node: ast.AST) -> None:
        self._loads_from_scope(node)  # comprehension vars do not leak

    visit_ListComp = visit_SetComp = visit_DictComp = _visit_comp
    visit_GeneratorExp = _visit_comp

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name.split(".")[0], value=False)

    visit_ImportFrom = visit_Import

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self._bind(node.name, value=False)  # scoped, unbound after handler
        for stmt in node.body:
            self.visit(stmt)


def analyze_names(code: str) -> tuple[list[str], list[str]]:
    """Detect the transfer-set candidates of a code block.

    Returns ``(free, assigned)``, both in first-appearance order:

    - *free* — names loaded before any top-level binding (candidate inputs);
      builtins excluded. Includes free names of nested scopes, counted at
      the definition site.
    - *assigned* — names value-bound at the top level (candidate outputs);
      imports, def/class names and ``_``-prefixed names excluded.

    Adapters intersect *free* with the live namespace and apply their own
    transfer guards; *assigned* pairs with ``lenient_outputs=True`` so
    non-transferable values are dropped remotely instead of failing the task.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SerializationError(f"code block does not parse: {exc}") from exc
    scan = _NameScan()
    scan.visit(tree)
    free = [
        n
        for n, pos in sorted(scan.first_load.items(), key=lambda kv: kv[1])
        if n not in _BUILTIN_NAMES
        and (n not in scan.first_store or pos < scan.first_store[n])
    ]
    assigned = [
        n
        for n, _ in sorted(scan.value_store.items(), key=lambda kv: kv[1])
        if not n.startswith("_")
    ]
    return free, assigned


def build_code_source(
    code: str,
    inputs: list[str],
    outputs: list[str],
    *,
    lenient_outputs: bool = False,
) -> tuple[str, str]:
    """Synthesize ``(code_string, entry_point)`` for a code block.

    With ``lenient_outputs=True`` the generated function returns only the
    outputs that are actually set and JSON-safe (for auto-detected outputs);
    the default strict epilogue fails remotely on a missing/unsafe name.
    """
    _reject_unsupported(code)

    params = ", ".join(f"{n}=None" for n in inputs)
    body = textwrap.indent(code.rstrip() + "\n", "    ")

    if lenient_outputs:
        epilogue = _EPILOGUE_LENIENT.format(names=list(outputs))
    else:
        returns = ", ".join(f"{n!r}: {n}" for n in outputs)
        epilogue = _EPILOGUE.format(returns=returns)

    source = (
        _PROLOGUE.format(params=params)
        + body
        + epilogue
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
