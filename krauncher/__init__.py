# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""krauncher — CaS client library for submitting GPU compute tasks.

Usage::

    from krauncher import KrauncherClient

    client = KrauncherClient(api_key="cas_...", broker_url="http://...")

    @client.task(vram_gb=24, timeout=3600)
    def train(data):
        import torch
        return {"loss": 0.01}

    async def main():
        handle = await train(data={"epochs": 5})
        result = await handle
        print(result.output)
"""

import logging as _logging
import os as _os

# Load .env from CWD before anything else (does NOT override existing vars)
from ._env import load_dotenv as _load_dotenv
_load_dotenv()

_log = _logging.getLogger("krauncher")
if not _log.handlers:
    _handler = _logging.StreamHandler()
    _handler.setFormatter(_logging.Formatter("%(message)s"))
    _log.addHandler(_handler)
if _os.getenv("KRAUNCHER_DEBUG", "").lower() in ("1", "true", "yes"):
    _log.setLevel(_logging.DEBUG)
else:
    _log.setLevel(_logging.INFO)

from .exceptions import (
    AuthError,
    InsufficientBalanceError,
    KrauncherError,
    NoCapacityError,
    PayloadDeliveryError,
    RemoteTimeout,
    SerializationError,
    TaskError,
    TaskTimeout,
    ValueTransferError,
)
from .analyzer import TaskClassification
from .data_source import DataSource
from .KrauncherClient import KrauncherClient
from .models import Runner, TaskHandle, TaskResult
from .volume import Volume

__all__ = [
    "KrauncherClient",
    "DataSource",
    "Volume",
    "TaskClassification",
    "Runner",
    "TaskHandle",
    "TaskResult",
    "KrauncherError",
    "AuthError",
    "InsufficientBalanceError",
    "TaskError",
    "TaskTimeout",
    "NoCapacityError",
    "PayloadDeliveryError",
    "RemoteTimeout",
    "SerializationError",
    "ValueTransferError",
]

__version__ = "0.1.0"
