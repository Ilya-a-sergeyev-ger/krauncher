# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for krauncher.volume — directory download mirrors directory upload."""

from krauncher.volume import Volume


class _StubVolume(Volume):
    """Volume with the two HTTP calls replaced by recording stubs."""

    def __init__(self, keys):
        self.name = "vol"
        self._keys = keys
        self.downloaded: list[tuple[str, str]] = []

    def ls(self, prefix: str = ""):
        return [{"key": k} for k in self._keys if k.startswith(prefix)]

    def download(self, remote_path: str, local_path: str) -> None:
        self.downloaded.append((remote_path, local_path))


def test_download_dir_keeps_relative_layout(tmp_path):
    vol = _StubVolume(["out/a.png", "out/nested/b.png"])

    assert vol.download_dir("out", str(tmp_path)) == 2
    assert vol.downloaded == [
        ("out/a.png", str(tmp_path / "a.png")),
        ("out/nested/b.png", str(tmp_path / "nested/b.png")),
    ]


def test_prefix_matches_a_directory_not_a_string(tmp_path):
    # "output.txt" also starts with "out" — a directory download must not
    # sweep it in.
    vol = _StubVolume(["out/a.png", "output.txt"])

    assert vol.download_dir("out", str(tmp_path)) == 1
    assert vol.downloaded == [("out/a.png", str(tmp_path / "a.png"))]


def test_empty_prefix_takes_the_whole_volume(tmp_path):
    vol = _StubVolume(["a.png", "sub/b.png"])

    assert vol.download_dir("", str(tmp_path)) == 2
    assert [remote for remote, _ in vol.downloaded] == ["a.png", "sub/b.png"]


def test_directory_placeholder_keys_are_skipped(tmp_path):
    vol = _StubVolume(["out/", "out/a.png"])

    assert vol.download_dir("out", str(tmp_path)) == 1
    assert vol.downloaded == [("out/a.png", str(tmp_path / "a.png"))]


def test_leading_and_trailing_slashes_are_tolerated(tmp_path):
    vol = _StubVolume(["out/a.png"])

    assert vol.download_dir("/out/", str(tmp_path)) == 1
    assert vol.downloaded == [("out/a.png", str(tmp_path / "a.png"))]
