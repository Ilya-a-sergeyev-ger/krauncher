"""KrauncherClient.ladder(): broker /v1/ladder, 409 -> AssayOutdated, 403 -> KrauncherError."""
import asyncio

import httpx
import pytest

from krauncher import KrauncherClient, KrauncherError
from krauncher.analyzer import AssayOutdated


def _client(monkeypatch, status, body):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["key"] = str(request.url), request.headers.get("X-API-Key")
        return httpx.Response(status, json=body)

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, transport=httpx.MockTransport(handler), **kw))
    return KrauncherClient(api_key="k", broker_url="https://broker.test/api"), seen


def test_ok(monkeypatch):
    c, seen = _client(monkeypatch, 200, {"rows": [{"gpu_id": "x"}]})
    assert asyncio.run(c.ladder({"meta": {}}))["rows"] == [{"gpu_id": "x"}]
    assert seen == {"url": "https://broker.test/api/v1/ladder", "key": "k"}


def test_outdated(monkeypatch):
    c, _ = _client(monkeypatch, 409, {"detail": {"assay_calibration_id": None}})
    with pytest.raises(AssayOutdated):
        asyncio.run(c.ladder({}))


def test_no_access(monkeypatch):
    c, _ = _client(monkeypatch, 403, {"detail": "Ladder access is not enabled for this account"})
    with pytest.raises(KrauncherError):
        asyncio.run(c.ladder({}))
