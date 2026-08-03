# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Minimal .env loader for krauncher.

Reads key=value pairs from a .env file into os.environ.
Does NOT override existing environment variables.

With no explicit path, the file is resolved from the KRAUNCHER_CLIENT_CONFIG
environment variable (a config filename, e.g. set once in a notebook so
secrets stay out of the cells) if present, otherwise ``.env`` in the CWD.
It must be a real env var — it cannot live inside the file it points to.

Settings are read through ``setting()`` below, which accepts the KRAUNCHER_
prefix and the original CAS_ one.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> int:
    """Load .env file into os.environ. Returns number of vars set."""
    if path is None:
        config = (os.environ.get("KRAUNCHER_CLIENT_CONFIG")
                  or os.environ.get("CAS_CLIENT_CONFIG"))
        path = Path(config) if config else Path.cwd() / ".env"
    else:
        path = Path(path)

    if not path.is_file():
        return 0

    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        # Don't override existing env vars
        if key not in os.environ:
            os.environ[key] = value
            count += 1

    return count


def setting(name: str, default: str = "") -> str:
    """Read one krauncher setting by its bare name (e.g. ``API_KEY``).

    ``KRAUNCHER_<NAME>`` is the spelling to use. ``CAS_<NAME>`` is the original
    one and keeps working: .env files, notebooks and CI in the wild are full of
    CAS_API_KEY, and a rename that breaks them buys nothing.

    Both are read from ``os.environ``, which ``load_dotenv`` has already filled
    from the .env file.
    """
    value = os.environ.get(f"KRAUNCHER_{name}")
    if value is None:
        value = os.environ.get(f"CAS_{name}")
    return default if value is None else value
