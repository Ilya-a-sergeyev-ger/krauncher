# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""S3 reference detection and rewrite for code blocks.

Unlike the HF path there is no cache layer to make pre-fetched data visible
to unmodified code — plain ``pd.read_csv("s3://...")`` goes to S3 again at
execution time (inside the billed compute phase, and without credentials in
the task env). The loop is closed by rewriting: detected literal object URLs
are pre-fetched by the data bridge and the literals are rewritten to the
local mount path (``/data/<basename>``) in the synthesized source, so the
code reads the local copy and the IO stays in the measured download phase.

Only whole-literal exact object URLs translate. Prefixes (``.../dir/``) and
dynamic constructions (f-strings) are reported back so the adapter can warn.
"""

from __future__ import annotations

import ast
import posixpath
import re
from collections import Counter
from urllib.parse import urlparse


def detect_s3_refs(code: str) -> tuple[list[str], list[str]]:
    """Detect ``s3://`` references in a code block.

    Returns ``(urls, notes)``: exact-object URLs ready for translation
    (first-appearance order, deduplicated) and human-readable notes about
    references that cannot be translated (prefixes, f-strings).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], []

    # Constants inside f-strings are dynamic URLs — note, don't translate.
    fstring_parts: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for part in ast.walk(node):
                if isinstance(part, ast.Constant):
                    fstring_parts.add(id(part))

    urls: list[str] = []
    notes: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        value = node.value
        if not value.startswith("s3://") or value in seen:
            continue
        seen.add(value)
        if id(node) in fstring_parts:
            notes.append(
                f"{value}...: dynamic s3 reference (f-string) — downloads "
                f"in-code, IO will be billed as compute"
            )
        elif value.endswith("/"):
            notes.append(
                f"{value}: s3 prefix — not translated (pre-fetch handles "
                f"single objects); downloads in-code if read"
            )
        else:
            urls.append(value)
    return urls, notes


def s3_local_mapping(urls: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map exact object URLs to their data-bridge mount paths.

    Mirrors the worker's dest naming (``/data/<basename of the key>``).
    Basename collisions and keyless URLs are excluded with a note — a wrong
    silent rewrite is worse than an untranslated one.
    """
    base_of = {u: posixpath.basename(urlparse(u).path) for u in urls}
    counts = Counter(base_of.values())
    mapping: dict[str, str] = {}
    notes: list[str] = []
    for url, base in base_of.items():
        if not base:
            notes.append(f"{url}: no object key — not translated")
        elif counts[base] > 1:
            notes.append(f"{url}: filename collision with another URL — not translated")
        else:
            mapping[url] = f"/data/{base}"
    return mapping, notes


def rewrite_s3_refs(code: str, mapping: dict[str, str]) -> str:
    """Rewrite whole string literals holding translated URLs to local paths.

    Only quoted whole-literal occurrences are touched (``"s3://..."`` /
    ``'s3://...'``) — exactly the shape :func:`detect_s3_refs` translates.
    """
    for url in sorted(mapping, key=len, reverse=True):
        local = mapping[url]
        code = re.sub(
            "(['\"])" + re.escape(url) + r"\1",
            lambda m, p=local: m.group(1) + p + m.group(1),
            code,
        )
    return code
