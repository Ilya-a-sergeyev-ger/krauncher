# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""HuggingFace reference detection and sizing for code blocks.

Translates the established notebook practice — ``load_dataset("org/name")``,
``AutoModel.from_pretrained("org/name")`` — into ``hf://`` data-bridge URLs,
so the download happens *before* the container starts: the IO lands in the
measured download phase (honest cu_io / disk sizing, not billed as compute)
and the hub-cache layout (``#layout=cache``) lets the unmodified user code
find the pre-downloaded snapshot through ``HF_HOME``.

Only literal repo ids translate; dynamic references are reported back so the
adapter can warn that their IO will run inside execution.
"""

from __future__ import annotations

import ast
from typing import Any

import httpx

# load_dataset() first args that are packaged builders, not Hub repos.
_DATASET_BUILDERS = frozenset({
    "json", "csv", "parquet", "text", "pandas", "arrow",
    "imagefolder", "audiofolder", "videofolder", "webdataset", "sql",
})

# Fragment appended to emitted URLs: hub-cache layout on the worker.
CACHE_FRAGMENT = "#layout=cache"


_SKIP = object()  # literal, but not a Hub repo (local path) — ignore silently


def _literal_repo_id(node: ast.Call):
    """First positional arg: repo id str, _SKIP (local path), or None (dynamic)."""
    if not node.args:
        return None
    arg = node.args[0]
    if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
        return None
    repo_id = arg.value
    if repo_id.startswith((".", "/", "~")):
        return _SKIP
    return repo_id


def detect_hf_refs(code: str) -> tuple[list[str], list[str]]:
    """Detect HuggingFace Hub references in a code block.

    Returns ``(urls, dynamic)``:

    - *urls* — ``hf://datasets/...`` / ``hf://models/...`` for literal repo
      ids (first-appearance order, deduplicated), ready for ``data_urls``;
    - *dynamic* — human-readable descriptions of Hub calls whose repo id is
      not a literal (cannot be pre-fetched).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], []

    urls: list[str] = []
    dynamic: list[str] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "load_dataset":
            kind = "datasets"
        elif isinstance(func, ast.Attribute) and func.attr == "from_pretrained":
            kind = "models"
        else:
            continue

        repo_id = _literal_repo_id(node)
        if repo_id is _SKIP:
            continue  # literal local path — nothing to pre-fetch
        if repo_id is None:
            name = func.id if isinstance(func, ast.Name) else f"...{func.attr}"
            dynamic.append(f"{name}(...)")
            continue
        if kind == "datasets" and repo_id in _DATASET_BUILDERS:
            continue  # packaged builder (local files), not a Hub repo

        url = f"hf://{kind}/{repo_id}"
        if url not in seen:
            seen.add(url)
            urls.append(url)

    return urls, dynamic


async def hf_size_mb(urls: list[str], *, timeout: float = 5.0) -> float | None:
    """Total size (MB) of ``hf://`` repos via the public Hub API, best-effort.

    Feeds the analysis phase (cu_io / disk) before anything is submitted.
    Returns None when nothing could be resolved (private repo, network).
    """
    total = 0
    resolved = False
    async with httpx.AsyncClient(timeout=timeout) as session:
        for url in urls:
            path = url.removeprefix("hf://").split("#", 1)[0].strip("/")
            parts = path.split("/")
            if parts[0] in ("datasets", "models"):
                repo_type, repo_id = parts[0], "/".join(parts[1:3])
            else:
                repo_type, repo_id = "models", "/".join(parts[:2])
            try:
                resp = await session.get(
                    f"https://huggingface.co/api/{repo_type}/{repo_id}",
                    params={"blobs": "true"},
                )
                if resp.status_code != 200:
                    continue
                siblings: list[dict[str, Any]] = resp.json().get("siblings") or []
                total += sum(s.get("size") or 0 for s in siblings)
                resolved = True
            except Exception:
                continue
    return total / (1 << 20) if resolved else None
