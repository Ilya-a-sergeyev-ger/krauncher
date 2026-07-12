# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""E2E encryption utilities for krauncher (cas-client).

Mirrors cas-worker/src/relay/crypto.py — same algorithm, same wire format.

Wire format: base64url(nonce_12B || ciphertext_with_16B_tag)
"""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_HKDF_INFO = b"cas-e2e-v1"


def _public_bytes_raw(pub: X25519PublicKey) -> bytes:
    """Return the 32-byte raw public key, compatible with old cryptography.

    `public_bytes_raw()` exists since cryptography 40.0 (Apr 2023). On older
    runtimes (some sandbox/provider images still ship pre-40), fall back to
    the verbose API.
    """
    try:
        return pub.public_bytes_raw()
    except AttributeError:
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
        )
        return pub.public_bytes(Encoding.Raw, PublicFormat.Raw)


def generate_keypair() -> tuple[X25519PrivateKey, bytes]:
    """Generate an ephemeral X25519 keypair.

    Returns:
        (private_key, public_key_bytes) — public key is 32 raw bytes.
    """
    priv = X25519PrivateKey.generate()
    pub_bytes = _public_bytes_raw(priv.public_key())
    return priv, pub_bytes


def derive_shared_secret(
    private_key: X25519PrivateKey,
    peer_public_bytes: bytes,
) -> bytes:
    """Derive a 32-byte symmetric key via X25519 + HKDF-SHA256."""
    peer_pub = X25519PublicKey.from_public_bytes(peer_public_bytes)
    dh_secret = private_key.exchange(peer_pub)
    return HKDF(
        algorithm=SHA256(),
        length=32,
        salt=None,
        info=_HKDF_INFO,
    ).derive(dh_secret)


def encrypt(key: bytes, plaintext: bytes) -> str:
    """Encrypt *plaintext* and return base64url(nonce || ciphertext)."""
    nonce = os.urandom(12)
    ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, None)
    return base64.urlsafe_b64encode(nonce + ciphertext).decode()


def decrypt(key: bytes, encoded: str) -> bytes:
    """Decrypt a value produced by :func:`encrypt`."""
    raw = base64.urlsafe_b64decode(encoded + "==")
    if len(raw) < 12:
        raise ValueError(f"encrypted blob too short: {len(raw)} bytes")
    nonce, ciphertext = raw[:12], raw[12:]
    return ChaCha20Poly1305(key).decrypt(nonce, ciphertext, None)


# ---------------------------------------------------------------------------
# Sealed box for stored credentials (fleet key)
# ---------------------------------------------------------------------------
# Encrypt-at-source for storage credentials: the browser / client seals to
# the fleet public key; broker, MariaDB and RabbitMQ carry ciphertext only;
# the worker unseals with the fleet private key right before use.
# Wire format: base64url( eph_pub(32) || nonce(12) || chacha20poly1305_ct ).

_CREDS_HKDF_INFO = b"cas-creds-v1"


def _creds_key(dh_secret: bytes) -> bytes:
    return HKDF(
        algorithm=SHA256(),
        length=32,
        salt=None,
        info=_CREDS_HKDF_INFO,
    ).derive(dh_secret)


def seal(peer_public_b64: str, plaintext: bytes) -> str:
    """Seal *plaintext* to a base64url-encoded X25519 public key."""
    peer_pub_bytes = base64.urlsafe_b64decode(peer_public_b64 + "==")
    eph_priv, eph_pub = generate_keypair()
    key = _creds_key(
        eph_priv.exchange(X25519PublicKey.from_public_bytes(peer_pub_bytes))
    )
    nonce = os.urandom(12)
    ct = ChaCha20Poly1305(key).encrypt(nonce, plaintext, None)
    return base64.urlsafe_b64encode(eph_pub + nonce + ct).decode()


def unseal(private_key_b64: str, blob: str) -> bytes:
    """Open a blob produced by :func:`seal` with the fleet private key."""
    priv = X25519PrivateKey.from_private_bytes(
        base64.urlsafe_b64decode(private_key_b64 + "==")
    )
    raw = base64.urlsafe_b64decode(blob + "==")
    if len(raw) < 32 + 12:
        raise ValueError(f"sealed blob too short: {len(raw)} bytes")
    eph_pub, nonce, ct = raw[:32], raw[32:44], raw[44:]
    key = _creds_key(priv.exchange(X25519PublicKey.from_public_bytes(eph_pub)))
    return ChaCha20Poly1305(key).decrypt(nonce, ct, None)
