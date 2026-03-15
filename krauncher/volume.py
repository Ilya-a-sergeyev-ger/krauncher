"""Volume — manage persistent S3-backed volumes on the broker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .KrauncherClient import KrauncherClient

from .models import _check_response


class Volume:
    """A persistent S3-backed volume on the CaS broker.

    Wraps the ``/volumes`` REST endpoints.  Creating an instance will
    ensure the volume exists (create if missing, no-op if exists).

    Usage::

        vol = client.volume("checkpoints", size_gb=10)
        vol.upload("./local_model.pt", "models/model.pt")
        vol.download("models/model.pt", "./downloaded.pt")
        print(vol.ls())
        vol.delete()
    """

    def __init__(
        self,
        client: KrauncherClient,
        name: str,
        size_gb: int = 5,
    ) -> None:
        self._client = client
        self.name = name
        self.size_gb = size_gb
        self._ensure_exists()

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._client.api_key}

    def _url(self, path: str = "") -> str:
        return f"{self._client.broker_url}/volumes{path}"

    def _ensure_exists(self) -> None:
        """Create volume if it doesn't exist yet."""
        with httpx.Client(timeout=10.0) as session:
            resp = session.get(self._url(f"/{self.name}"), headers=self._headers())
            if resp.status_code == 200:
                return  # already exists
            # Create
            resp = session.post(
                self._url(),
                json={"name": self.name, "size_gb": self.size_gb},
                headers=self._headers(),
            )
            _check_response(resp)

    @property
    def info(self) -> dict[str, Any]:
        """Fetch volume details (size, used_bytes, etc.)."""
        with httpx.Client(timeout=10.0) as session:
            resp = session.get(self._url(f"/{self.name}"), headers=self._headers())
            _check_response(resp)
            return resp.json()

    def ls(self, prefix: str = "") -> list[dict[str, Any]]:
        """List files in the volume."""
        with httpx.Client(timeout=30.0) as session:
            resp = session.get(
                self._url(f"/{self.name}/files"),
                headers=self._headers(),
            )
            _check_response(resp)
            files = resp.json().get("files", [])
            if prefix:
                files = [f for f in files if f.get("key", "").startswith(prefix)]
            return files

    def upload(self, local_path: str, dest: str = "") -> int:
        """Upload a file or directory to the volume via presigned URLs.

        Args:
            local_path: Local file or directory path.
            dest: Destination path inside the volume. If empty, uses the
                filename (for files) or preserves relative paths (for dirs).

        Returns:
            Number of files uploaded.
        """
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Local path not found: {local_path}")

        if src.is_file():
            remote_path = dest if dest else src.name
            self._upload_file(src, remote_path)
            return 1

        # Directory: walk and upload each file
        count = 0
        for file_path in src.rglob("*"):
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(src))
            remote_path = f"{dest}/{rel}" if dest else rel
            self._upload_file(file_path, remote_path)
            count += 1
        return count

    def _upload_file(self, local_file: Path, remote_path: str) -> None:
        """Upload a single file via presigned PUT URL."""
        remote_path = remote_path.lstrip("/")
        with httpx.Client(timeout=60.0) as session:
            # Get presigned URL
            resp = session.post(
                self._url(f"/{self.name}/presign"),
                json={
                    "operation": "upload",
                    "path": remote_path,
                },
                headers=self._headers(),
            )
            _check_response(resp)
            presigned_url = resp.json()["url"]

            # Upload
            with open(local_file, "rb") as f:
                put_resp = session.put(presigned_url, content=f.read())
                put_resp.raise_for_status()

    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file from the volume via presigned GET URL.

        Args:
            remote_path: Path inside the volume.
            local_path: Local destination file path.
        """
        remote_path = remote_path.lstrip("/")
        with httpx.Client(timeout=60.0) as session:
            resp = session.post(
                self._url(f"/{self.name}/presign"),
                json={
                    "operation": "download",
                    "path": remote_path,
                },
                headers=self._headers(),
            )
            _check_response(resp)
            presigned_url = resp.json()["url"]

            # Download
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with session.stream("GET", presigned_url) as dl:
                dl.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in dl.iter_bytes(chunk_size=8192):
                        f.write(chunk)

    def delete(self) -> None:
        """Delete this volume and all its data."""
        with httpx.Client(timeout=30.0) as session:
            resp = session.delete(
                self._url(f"/{self.name}"),
                headers=self._headers(),
            )
            _check_response(resp)
