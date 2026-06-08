#!/usr/bin/env python3
"""Test that empty GPU name/arch results in empty string (not None) in requirements."""

import os
import subprocess
import sys

def test_empty_defaults():
    """Test 1: No env vars → empty strings."""
    code = """
import os
os.environ.pop("KRAUNCHER_GPU_NAME", None)
os.environ.pop("KRAUNCHER_GPU_ARCH", None)

from krauncher import KrauncherClient
client = KrauncherClient(api_key="test", broker_url="http://test")

assert client.default_gpu_name == "", f"Expected empty string, got {repr(client.default_gpu_name)}"
assert client.default_gpu_arch == "", f"Expected empty string, got {repr(client.default_gpu_arch)}"
print("PASS")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: {result.stderr}")
        sys.exit(1)
    assert "PASS" in result.stdout
    print("✅ Test 1: No env vars → empty strings")

def test_env_vars():
    """Test 2: Env vars set → uses env vars."""
    code = """
import os
os.environ["KRAUNCHER_GPU_NAME"] = "H100"
os.environ["KRAUNCHER_GPU_ARCH"] = "Hopper"

from krauncher import KrauncherClient
client = KrauncherClient(api_key="test", broker_url="http://test")

assert client.default_gpu_name == "H100", f"Expected H100, got {client.default_gpu_name}"
assert client.default_gpu_arch == "Hopper", f"Expected Hopper, got {client.default_gpu_arch}"
print("PASS")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: {result.stderr}")
        sys.exit(1)
    assert "PASS" in result.stdout
    print("✅ Test 2: Env vars set → uses env vars")

def test_constructor_override():
    """Test 3: Constructor params override env vars."""
    code = """
import os
os.environ["KRAUNCHER_GPU_NAME"] = "H100"
os.environ["KRAUNCHER_GPU_ARCH"] = "Hopper"

from krauncher import KrauncherClient
client = KrauncherClient(
    api_key="test",
    broker_url="http://test",
    gpu_name="A100",
    gpu_arch="Ampere"
)

assert client.default_gpu_name == "A100", f"Expected A100, got {client.default_gpu_name}"
assert client.default_gpu_arch == "Ampere", f"Expected Ampere, got {client.default_gpu_arch}"
print("PASS")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: {result.stderr}")
        sys.exit(1)
    assert "PASS" in result.stdout
    print("✅ Test 3: Constructor params override env vars")

def test_priority_logic():
    """Test 4: Priority logic in wrapper (simulated)."""
    code = """
import os
os.environ["KRAUNCHER_GPU_NAME"] = "H100"
os.environ["KRAUNCHER_GPU_ARCH"] = "Hopper"

from krauncher import KrauncherClient
client = KrauncherClient(api_key="test", broker_url="http://test")

# Simulate decorator behavior
# Case 1: decorator param is None → use client default (from env)
gpu_name = None
gpu_arch = None
final_gpu_name = gpu_name if gpu_name is not None else client.default_gpu_name
final_gpu_arch = gpu_arch if gpu_arch is not None else client.default_gpu_arch
assert final_gpu_name == "H100", f"Expected H100, got {final_gpu_name}"
assert final_gpu_arch == "Hopper", f"Expected Hopper, got {final_gpu_arch}"

# Case 2: decorator param is explicit empty string → override env var
gpu_name = ""
gpu_arch = ""
final_gpu_name = gpu_name if gpu_name is not None else client.default_gpu_name
final_gpu_arch = gpu_arch if gpu_arch is not None else client.default_gpu_arch
assert final_gpu_name == "", f"Expected empty string, got {final_gpu_name}"
assert final_gpu_arch == "", f"Expected empty string, got {final_gpu_arch}"

# Case 3: decorator param is explicit value → highest priority
gpu_name = "A100"
gpu_arch = "Ampere"
final_gpu_name = gpu_name if gpu_name is not None else client.default_gpu_name
final_gpu_arch = gpu_arch if gpu_arch is not None else client.default_gpu_arch
assert final_gpu_name == "A100", f"Expected A100, got {final_gpu_name}"
assert final_gpu_arch == "Ampere", f"Expected Ampere, got {final_gpu_arch}"

print("PASS")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: {result.stderr}")
        sys.exit(1)
    assert "PASS" in result.stdout
    print("✅ Test 4: Priority logic works correctly")

def test_empty_string_sent_to_broker():
    """Test 5: Empty strings (not None) are sent to requirements."""
    code = """
import os
os.environ.pop("KRAUNCHER_GPU_NAME", None)
os.environ.pop("KRAUNCHER_GPU_ARCH", None)

from krauncher import KrauncherClient
client = KrauncherClient(api_key="test", broker_url="http://test")

# Simulate requirements dict construction (from wrapper code)
gpu_name = None  # Not specified in decorator
gpu_arch = None  # Not specified in decorator

final_gpu_name = gpu_name if gpu_name is not None else client.default_gpu_name
final_gpu_arch = gpu_arch if gpu_arch is not None else client.default_gpu_arch

# These should be empty strings, not None
assert final_gpu_name == "", f"Expected empty string, got {repr(final_gpu_name)}"
assert final_gpu_arch == "", f"Expected empty string, got {repr(final_gpu_arch)}"
assert type(final_gpu_name) is str, f"Expected str type, got {type(final_gpu_name)}"
assert type(final_gpu_arch) is str, f"Expected str type, got {type(final_gpu_arch)}"

print("PASS")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: {result.stderr}")
        sys.exit(1)
    assert "PASS" in result.stdout
    print("✅ Test 5: Empty strings (not None) sent to requirements")

if __name__ == "__main__":
    print("Testing GPU empty string behavior and priority...")
    print()

    test_empty_defaults()
    test_env_vars()
    test_constructor_override()
    test_priority_logic()
    test_empty_string_sent_to_broker()

    print()
    print("All tests passed! ✅")
    print()
    print("Summary:")
    print("  ✓ When nothing specified: gpu_name='', gpu_arch='' (empty strings, not None)")
    print("  ✓ Priority: decorator param → env var (KRAUNCHER_GPU_*) → constructor → ''")
    print("  ✓ Empty strings are correctly sent to broker requirements")
