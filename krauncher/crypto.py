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


def generate_keypair() -> tuple[X25519PrivateKey, bytes]:
    """Generate an ephemeral X25519 keypair.

    Returns:
        (private_key, public_key_bytes) — public key is 32 raw bytes.
    """
    priv = X25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes_raw()
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
