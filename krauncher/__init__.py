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

from .exceptions import (
    AuthError,
    KrauncherError,
    RemoteTimeout,
    SerializationError,
    TaskError,
    TaskTimeout,
)
from .KrauncherClient import KrauncherClient
from .models import TaskHandle, TaskResult

__all__ = [
    "KrauncherClient",
    "TaskHandle",
    "TaskResult",
    "KrauncherError",
    "AuthError",
    "TaskError",
    "TaskTimeout",
    "RemoteTimeout",
    "SerializationError",
]

__version__ = "0.1.0"
