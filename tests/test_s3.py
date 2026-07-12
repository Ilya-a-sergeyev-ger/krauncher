# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Tests for S3 reference detection and rewrite (krauncher.s3)."""

from krauncher.s3 import detect_s3_refs, rewrite_s3_refs, s3_local_mapping


def test_detect_exact_object():
    urls, notes = detect_s3_refs('df = pd.read_csv("s3://bucket/data.csv")\n')
    assert urls == ["s3://bucket/data.csv"]
    assert notes == []


def test_detect_prefix_noted_not_translated():
    urls, notes = detect_s3_refs('fs.ls("s3://bucket/dir/")\n')
    assert urls == []
    assert any("prefix" in n for n in notes)


def test_detect_fstring_noted():
    urls, notes = detect_s3_refs('p = f"s3://bucket/{name}.csv"\n')
    assert urls == []
    assert any("f-string" in n for n in notes)


def test_detect_dedup_order():
    code = (
        'a = open("s3://b/one.bin")\n'
        'b = open("s3://b/two.bin")\n'
        'c = open("s3://b/one.bin")\n'
    )
    urls, _ = detect_s3_refs(code)
    assert urls == ["s3://b/one.bin", "s3://b/two.bin"]


def test_mapping_matches_bridge_naming():
    mapping, notes = s3_local_mapping(["s3://bucket/dir/data.csv"])
    assert mapping == {"s3://bucket/dir/data.csv": "/data/data.csv"}
    assert notes == []


def test_mapping_collision_excluded():
    mapping, notes = s3_local_mapping(["s3://a/x/f.csv", "s3://b/y/f.csv"])
    assert mapping == {}
    assert len(notes) == 2 and all("collision" in n for n in notes)


def test_mapping_keyless_excluded():
    mapping, notes = s3_local_mapping(["s3://bucket"])
    assert mapping == {}
    assert any("no object key" in n for n in notes)


def test_rewrite_both_quote_styles():
    code = "a = pd.read_csv(\"s3://b/d.csv\")\nb = open('s3://b/d.csv')\n"
    out = rewrite_s3_refs(code, {"s3://b/d.csv": "/data/d.csv"})
    assert '"/data/d.csv"' in out and "'/data/d.csv'" in out
    assert "s3://" not in out


def test_rewrite_leaves_longer_urls_alone():
    code = 'a = open("s3://b/k")\nb = open("s3://b/k2")\n'
    out = rewrite_s3_refs(code, {"s3://b/k": "/data/k"})
    assert '"/data/k"' in out
    assert '"s3://b/k2"' in out  # not corrupted by the shorter mapping


def test_rewrite_only_whole_literals():
    code = 'print("see s3://b/d.csv for source")\n'
    out = rewrite_s3_refs(code, {"s3://b/d.csv": "/data/d.csv"})
    assert out == code  # embedded mention, not a whole literal — untouched
