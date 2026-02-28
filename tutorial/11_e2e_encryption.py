"""Tutorial 11: End-to-End Encryption.

Demonstrates that task code and arguments never reach the broker or RabbitMQ
in plaintext. Only metadata (vram_gb, timeout, pip deps) is visible to the
infrastructure; code and args travel encrypted via relay after key exchange.

Protocol:
  1. Client generates ephemeral X25519 keypair (ek_priv, ek_pub).
  2. POST /tasks sends only metadata + ek_pub; code_string is empty.
  3. Worker picks up task, generates its own keypair (wk_priv, wk_pub).
  4. Worker sends key_exchange event via relay with wk_pub.
  5. Client derives shared_secret = HKDF(X25519(ek_priv, wk_pub)).
  6. Worker derives shared_secret = HKDF(X25519(wk_priv, ek_pub)).
  7. Client encrypts {code_string, args} → PUT relay/tasks/{id}/payload.
  8. Worker decrypts payload, executes code.
  9. Relay stdout/stderr stream is also encrypted.

Usage:
    CAS_API_KEY=cas_... python -m krauncher.tutorial.11_e2e_encryption
"""

import asyncio
import os

from krauncher import KrauncherClient


API_KEY = os.environ.get("CAS_API_KEY", "")
BROKER_URL = os.environ.get("CAS_BROKER_URL", "http://localhost:8000")

_client = KrauncherClient(api_key=API_KEY, broker_url=BROKER_URL, encrypt=True)


@_client.task(vram_gb=4, timeout=120)
def secret_computation(secret_value: int, multiplier: int) -> dict:
    """This function body never appears in RabbitMQ or broker logs."""
    import hashlib
    result = secret_value * multiplier
    digest = hashlib.sha256(str(result).encode()).hexdigest()
    print(f"[worker] computed {secret_value} × {multiplier} = {result}")
    print(f"[worker] sha256({result}) = {digest}")
    return {"result": result, "digest": digest}


async def main() -> None:
    if not API_KEY:
        raise SystemExit("Set CAS_API_KEY environment variable")

    print("Submitting task with E2E encryption enabled...")
    print("  code_string is withheld from broker — only ek_pub is sent\n")

    logs: list[str] = []

    def on_log(msg: dict) -> None:
        t = msg.get("type", "")
        data = msg.get("data", {})
        if t in ("stdout", "stderr") and isinstance(data, dict):
            line = data.get("text", "").rstrip()
            if line:
                print(f"  [{t}] {line}")
                logs.append(line)
        elif t == "event" and isinstance(data, dict):
            name = data.get("name", "")
            if name not in ("stream_ended",):
                print(f"  [event] {name}")

    handle = await secret_computation(secret_value=12345, multiplier=67890)
    print(f"Task submitted: {handle.task_id}")
    print("Waiting for key_exchange and payload upload...\n")

    result = await handle.wait(on_log=on_log, timeout=180)

    print(f"\nTask completed: {result.status}")
    print(f"Output: {result.output}")

    assert result.status == "completed", f"Expected completed, got {result.status}"
    assert result.output["result"] == 12345 * 67890
    print("\nAll assertions passed — E2E encryption working correctly.")


if __name__ == "__main__":
    asyncio.run(main())
