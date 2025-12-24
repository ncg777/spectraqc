from __future__ import annotations
import hashlib
from spectraqc.utils.canonical_json import canonical_dumps


def sha256_hex_bytes(b: bytes) -> str:
    """Compute SHA256 hash of bytes and return hex string."""
    return hashlib.sha256(b).hexdigest()


def sha256_hex_canonical_json(obj) -> str:
    """Compute SHA256 hash of canonical JSON representation."""
    return sha256_hex_bytes(canonical_dumps(obj).encode("utf-8"))


def sha256_hex_file(path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
