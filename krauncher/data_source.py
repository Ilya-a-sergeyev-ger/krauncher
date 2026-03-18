# Copyright (c) 2024-2026 Ilya Sergeev. Licensed under the MIT License.

"""DataSource — manage registered data sources on the broker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .KrauncherClient import KrauncherClient

from .models import _check_response


class DataSource:
    """A registered data source on the CaS broker.

    Wraps the ``/data-sources`` REST endpoints.  Creating an instance with
    *urls* will register (or fail if duplicate) the data source on the broker.
    Creating without *urls* is a lightweight handle for an existing source.

    Usage::

        ds = client.data_source("imagenet", urls=["s3://bucket/data/"], size_gb=25)
        print(ds.info)
        ds.delete()
    """

    def __init__(
        self,
        client: KrauncherClient,
        name: str,
        urls: list[str] | None = None,
        size_gb: float = 0,
        description: str | None = None,
        is_output: bool = False,
    ) -> None:
        self._client = client
        self.name = name
        if urls is not None:
            self._create(urls, size_gb, description, is_output)

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._client.api_key}

    def _url(self, path: str = "") -> str:
        return f"{self._client.broker_url}/data-sources{path}"

    def _create(
        self,
        urls: list[str],
        size_gb: float,
        description: str | None,
        is_output: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": self.name,
            "urls": urls,
            "size_gb": size_gb,
            "is_output": is_output,
        }
        if description is not None:
            body["description"] = description
        with httpx.Client(timeout=30.0) as session:
            resp = session.post(self._url(), json=body, headers=self._headers())
            _check_response(resp)
            return resp.json()

    @property
    def info(self) -> dict[str, Any]:
        """Fetch data source details from the broker."""
        with httpx.Client(timeout=10.0) as session:
            resp = session.get(self._url(f"/{self.name}"), headers=self._headers())
            _check_response(resp)
            return resp.json()

    def delete(self) -> None:
        """Delete this data source."""
        with httpx.Client(timeout=10.0) as session:
            resp = session.delete(self._url(f"/{self.name}"), headers=self._headers())
            _check_response(resp)
