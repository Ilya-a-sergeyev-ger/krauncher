"""Tests for Phase 3: Runner model, @client.task(provider=...), list_runners().

Covers:
  - Runner dataclass: fields, __str__, status symbols
  - KrauncherClient.task(provider=...): provider_name in request body
  - KrauncherClient.list_runners(): fleet parsing, sorting, table output
"""

from __future__ import annotations

import json
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from krauncher import KrauncherClient, Runner
from krauncher.models import _STATUS_SYMBOL, TaskHandle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> KrauncherClient:
    return KrauncherClient(api_key="cas_test", broker_url="http://broker:8000")


def _fleet_response(hosts: list[dict], workers: list[dict] | None = None) -> dict:
    return {"hosts": hosts, "workers": workers or []}


def _host(host_id: str, provider: str, gpu: str = "A100 80GB", vram: int = 80,
          price: float = 2.5, status: str = "available", spot: bool = False) -> dict:
    return {
        "host_id": host_id,
        "provider_name": provider,
        "gpu_model": gpu,
        "gpu_count": 1,
        "vram_gb": vram,
        "gpu_arch": "Ampere",
        "price_per_hour_usd": price,
        "chunk_size_sec": 120,
        "status": status,
        "spot": spot,
        "region": "us-east",
    }


def _worker(worker_id: str, host_id: str, status: str = "idle") -> dict:
    return {"worker_id": worker_id, "host_id": host_id, "status": status}


def _mock_get(response_dict: dict, status_code: int = 200):
    """Return an async context manager mock that yields a response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = response_dict
    resp.text = json.dumps(response_dict)

    async_client = AsyncMock()
    async_client.__aenter__.return_value = async_client
    async_client.__aexit__.return_value = False
    async_client.get.return_value = resp
    async_client.post.return_value = resp
    return async_client


# ---------------------------------------------------------------------------
# 1. Runner model
# ---------------------------------------------------------------------------

class TestRunnerModel:
    def _runner(self, **kwargs) -> Runner:
        defaults = dict(
            provider="runpod", host_id="runpod-abc", gpu_model="A100 80GB",
            gpu_count=1, vram_gb=80, gpu_arch="Ampere",
            price_per_hour_usd=2.50, status="available",
            spot=False, region="us-east",
        )
        defaults.update(kwargs)
        return Runner(**defaults)

    def test_fields_stored_correctly(self):
        r = self._runner(provider="local", vram_gb=24)
        assert r.provider == "local"
        assert r.vram_gb == 24

    def test_worker_id_defaults_none(self):
        r = self._runner()
        assert r.worker_id is None

    def test_worker_id_set(self):
        r = self._runner(worker_id="w-001")
        assert r.worker_id == "w-001"

    def test_str_contains_provider_and_gpu(self):
        r = self._runner()
        s = str(r)
        assert "runpod" in s
        assert "A100 80GB" in s

    def test_str_shows_price(self):
        r = self._runner(price_per_hour_usd=2.50)
        assert "$2.50" in str(r)

    def test_str_shows_free_when_zero_price(self):
        r = self._runner(price_per_hour_usd=0.0)
        assert "free" in str(r)

    def test_str_shows_spot_tag(self):
        r = self._runner(spot=True)
        assert "spot" in str(r)

    def test_str_no_spot_tag_for_non_spot(self):
        r = self._runner(spot=False)
        assert "spot" not in str(r)

    def test_str_shows_status(self):
        r = self._runner(status="busy")
        assert "busy" in str(r)

    @pytest.mark.parametrize("status,symbol", list(_STATUS_SYMBOL.items()))
    def test_str_status_symbol(self, status, symbol):
        r = self._runner(status=status)
        assert str(r).startswith(symbol)

    def test_runner_is_frozen(self):
        r = self._runner()
        with pytest.raises((AttributeError, TypeError)):
            r.provider = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. @client.task(provider=...)
# ---------------------------------------------------------------------------

class TestTaskDecoratorProvider:
    """Tests that provider= is correctly forwarded into the HTTP request body."""

    def _make_wrapper(self, client: KrauncherClient, provider=None):
        """Create a decorated wrapper with serialize_function mocked out."""
        with patch("krauncher.KrauncherClient.serialize_function",
                   return_value=("def main(): pass", "main")):
            @client.task(vram_gb=24, provider=provider)
            def my_task(x):
                return x
        return my_task

    def test_provider_none_not_in_requirements(self):
        client = _make_client()
        captured: list[dict] = []

        async def fake_post(url, *, json=None, headers=None):
            captured.append(json or {})
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 201
            resp.json.return_value = {"task_id": "abc-123", "status": "queued"}
            return resp

        async def run():
            with patch("krauncher.KrauncherClient.serialize_function",
                       return_value=("def main(): pass", "main")):
                @client.task(vram_gb=16)
                def fn(x): return x

            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = False
            mock_session.post = fake_post

            with patch("krauncher.KrauncherClient.httpx.AsyncClient",
                       return_value=mock_session):
                await fn(x=1)

        import asyncio
        asyncio.get_event_loop().run_until_complete(run())

        assert captured
        reqs = captured[0].get("requirements", {})
        assert "provider_name" not in reqs or reqs.get("provider_name") is None

    def test_provider_runpod_in_requirements(self):
        client = _make_client()
        captured: list[dict] = []

        async def fake_post(url, *, json=None, headers=None):
            captured.append(json or {})
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 201
            resp.json.return_value = {"task_id": "abc-456", "status": "queued"}
            return resp

        async def run():
            with patch("krauncher.KrauncherClient.serialize_function",
                       return_value=("def main(): pass", "main")):
                @client.task(vram_gb=80, provider="runpod")
                def fn(x): return x

            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = False
            mock_session.post = fake_post

            with patch("krauncher.KrauncherClient.httpx.AsyncClient",
                       return_value=mock_session):
                await fn(x=1)

        import asyncio
        asyncio.get_event_loop().run_until_complete(run())

        assert captured
        reqs = captured[0].get("requirements", {})
        assert reqs.get("provider_name") == "runpod"

    def test_provider_local_in_requirements(self):
        client = _make_client()
        captured: list[dict] = []

        async def fake_post(url, *, json=None, headers=None):
            captured.append(json or {})
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 201
            resp.json.return_value = {"task_id": "abc-789", "status": "queued"}
            return resp

        async def run():
            with patch("krauncher.KrauncherClient.serialize_function",
                       return_value=("def main(): pass", "main")):
                @client.task(vram_gb=24, provider="local")
                def fn(x): return x

            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = False
            mock_session.post = fake_post

            with patch("krauncher.KrauncherClient.httpx.AsyncClient",
                       return_value=mock_session):
                await fn(x=1)

        import asyncio
        asyncio.get_event_loop().run_until_complete(run())

        reqs = captured[0].get("requirements", {})
        assert reqs.get("provider_name") == "local"

    def test_provider_metadata_stored(self):
        client = _make_client()
        with patch("krauncher.KrauncherClient.serialize_function",
                   return_value=("def main(): pass", "main")):
            @client.task(vram_gb=8, provider="runpod")
            def fn(x): return x

        assert fn._krauncher_provider == "runpod"

    def test_no_provider_metadata_none(self):
        client = _make_client()
        with patch("krauncher.KrauncherClient.serialize_function",
                   return_value=("def main(): pass", "main")):
            @client.task(vram_gb=8)
            def fn(x): return x

        assert fn._krauncher_provider is None

    def test_other_requirements_preserved_with_provider(self):
        client = _make_client()
        captured: list[dict] = []

        async def fake_post(url, *, json=None, headers=None):
            captured.append(json or {})
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 201
            resp.json.return_value = {"task_id": "abc-000", "status": "queued"}
            return resp

        async def run():
            with patch("krauncher.KrauncherClient.serialize_function",
                       return_value=("def main(): pass", "main")):
                @client.task(vram_gb=40, gpu_arch="Hopper", provider="runpod")
                def fn(x): return x

            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = False
            mock_session.post = fake_post

            with patch("krauncher.KrauncherClient.httpx.AsyncClient",
                       return_value=mock_session):
                await fn(x=1)

        import asyncio
        asyncio.get_event_loop().run_until_complete(run())

        reqs = captured[0]["requirements"]
        assert reqs["min_vram_gb"] == 40
        assert reqs["gpu_arch"] == "Hopper"
        assert reqs["provider_name"] == "runpod"


# ---------------------------------------------------------------------------
# 3. list_runners()
# ---------------------------------------------------------------------------

class TestListRunners:

    async def _call_list_runners(self, client, fleet: dict, print_table=False):
        """Call list_runners() with a mocked HTTP response."""
        mock_session = _mock_get(fleet)
        with patch("krauncher.KrauncherClient.httpx.AsyncClient",
                   return_value=mock_session):
            return await client.list_runners(print_table=print_table)

    @pytest.mark.asyncio
    async def test_returns_runner_list(self):
        client = _make_client()
        fleet = _fleet_response([_host("runpod-h1", "runpod")])
        runners = await self._call_list_runners(client, fleet)
        assert isinstance(runners, list)
        assert len(runners) == 1
        assert isinstance(runners[0], Runner)

    @pytest.mark.asyncio
    async def test_empty_fleet_returns_empty_list(self):
        client = _make_client()
        fleet = _fleet_response([])
        runners = await self._call_list_runners(client, fleet)
        assert runners == []

    @pytest.mark.asyncio
    async def test_runner_fields_populated(self):
        client = _make_client()
        fleet = _fleet_response([_host("runpod-h1", "runpod", vram=80, price=2.5,
                                       status="busy", spot=True)])
        runners = await self._call_list_runners(client, fleet)
        r = runners[0]
        assert r.provider == "runpod"
        assert r.host_id == "runpod-h1"
        assert r.vram_gb == 80
        assert r.price_per_hour_usd == 2.5
        assert r.status == "busy"
        assert r.spot is True

    @pytest.mark.asyncio
    async def test_worker_id_linked_to_host(self):
        client = _make_client()
        fleet = _fleet_response(
            hosts=[_host("runpod-h1", "runpod")],
            workers=[_worker("w-001", "runpod-h1")],
        )
        runners = await self._call_list_runners(client, fleet)
        assert runners[0].worker_id == "w-001"

    @pytest.mark.asyncio
    async def test_worker_id_none_when_no_worker(self):
        client = _make_client()
        fleet = _fleet_response([_host("runpod-h1", "runpod")])
        runners = await self._call_list_runners(client, fleet)
        assert runners[0].worker_id is None

    @pytest.mark.asyncio
    async def test_worker_not_matched_to_wrong_host(self):
        client = _make_client()
        fleet = _fleet_response(
            hosts=[_host("runpod-h1", "runpod"),
                   _host("runpod-h2", "runpod")],
            workers=[_worker("w-001", "runpod-h2")],
        )
        runners = await self._call_list_runners(client, fleet)
        by_id = {r.host_id: r for r in runners}
        assert by_id["runpod-h1"].worker_id is None
        assert by_id["runpod-h2"].worker_id == "w-001"

    @pytest.mark.asyncio
    async def test_local_sorted_before_runpod(self):
        client = _make_client()
        fleet = _fleet_response([
            _host("runpod-h1", "runpod"),
            _host("local-h1", "local", price=0.0),
        ])
        runners = await self._call_list_runners(client, fleet)
        assert runners[0].provider == "local"
        assert runners[1].provider == "runpod"

    @pytest.mark.asyncio
    async def test_multiple_providers_sorted(self):
        client = _make_client()
        fleet = _fleet_response([
            _host("runpod-h1", "runpod"),
            _host("local-h1", "local", price=0.0),
            _host("aws-h1", "aws"),
        ])
        runners = await self._call_list_runners(client, fleet)
        providers = [r.provider for r in runners]
        assert providers[0] == "local"
        # aws and runpod alphabetically after local
        assert set(providers[1:]) == {"aws", "runpod"}

    @pytest.mark.asyncio
    async def test_print_table_false_no_stdout(self, capsys):
        client = _make_client()
        fleet = _fleet_response([_host("runpod-h1", "runpod")])
        await self._call_list_runners(client, fleet, print_table=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    @pytest.mark.asyncio
    async def test_print_table_true_writes_stdout(self, capsys):
        client = _make_client()
        fleet = _fleet_response([_host("runpod-h1", "runpod")])
        await self._call_list_runners(client, fleet, print_table=True)
        captured = capsys.readouterr()
        assert "runpod" in captured.out
        assert "A100 80GB" in captured.out

    @pytest.mark.asyncio
    async def test_print_table_empty_fleet_no_crash(self, capsys):
        client = _make_client()
        fleet = _fleet_response([])
        await self._call_list_runners(client, fleet, print_table=True)
        captured = capsys.readouterr()
        assert "No runners" in captured.out

    @pytest.mark.asyncio
    async def test_uses_api_key_header(self):
        client = KrauncherClient(api_key="cas_secret", broker_url="http://broker:8000")
        fleet = _fleet_response([])

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = fleet
        mock_session.get.return_value = resp

        with patch("krauncher.KrauncherClient.httpx.AsyncClient",
                   return_value=mock_session):
            await client.list_runners(print_table=False)

        call_kwargs = mock_session.get.call_args.kwargs
        assert call_kwargs.get("headers", {}).get("X-API-Key") == "cas_secret"

    @pytest.mark.asyncio
    async def test_calls_admin_fleet_endpoint(self):
        client = _make_client()
        fleet = _fleet_response([])

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = fleet
        mock_session.get.return_value = resp

        with patch("krauncher.KrauncherClient.httpx.AsyncClient",
                   return_value=mock_session):
            await client.list_runners(print_table=False)

        url = mock_session.get.call_args.args[0]
        assert url.endswith("/admin/fleet")

    @pytest.mark.asyncio
    async def test_auth_error_raises(self):
        from krauncher.exceptions import AuthError

        client = _make_client()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 403
        resp.json.return_value = {"detail": "Forbidden"}
        resp.text = "Forbidden"

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        mock_session.get.return_value = resp

        with patch("krauncher.KrauncherClient.httpx.AsyncClient",
                   return_value=mock_session):
            with pytest.raises(AuthError):
                await client.list_runners(print_table=False)
