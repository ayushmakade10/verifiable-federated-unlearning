"""
evidence/hashing.py — Deterministic Model & File Hashing
=========================================================

Provides two hashing primitives used throughout the evidence bundle:

  1. hash_model(state_dict) — SHA-256 of a model's parameters.
     Serialises with sorted keys and raw tensor bytes so the hash
     is identical regardless of insertion order or platform.

  2. hash_file(path) — SHA-256 of an arbitrary file on disk.
     Used by the manifest to fingerprint every evidence artifact.

Specification reference: Section 6.3 of the master document:
  "Serialise state_dict() with sorted keys, concatenate raw tensor
   bytes, SHA-256 hash. Deterministic regardless of platform."

Usage:
    from evidence.hashing import hash_model, hash_file

    h = hash_model(model.state_dict())
    f = hash_file("outputs/run_001/final_model.pt")
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

import torch


def hash_model(state_dict: Dict[str, torch.Tensor]) -> str:
    """Compute a deterministic SHA-256 hash of a model's state_dict.

    Algorithm:
      1. Sort parameter keys lexicographically.
      2. For each key, move the tensor to CPU and convert to a
         contiguous numpy array, then extract raw bytes.
      3. Feed all byte buffers sequentially into a SHA-256 hasher.

    This produces an identical hash for the same weights regardless
    of GPU placement, parameter insertion order, or platform.

    Args:
        state_dict: The model's state_dict (from model.state_dict()
                    or loaded from a checkpoint).

    Returns:
        Lowercase hex digest string (64 characters).
    """
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        # Ensure CPU, contiguous layout, then raw bytes.
        raw = tensor.detach().cpu().numpy().tobytes()
        hasher.update(raw)
    return hasher.hexdigest()


def hash_file(path: str | Path) -> str:
    """Compute SHA-256 of a file's contents.

    Reads in 64 KB chunks to handle large checkpoint files without
    loading the entire file into memory.

    Args:
        path: Path to the file to hash.

    Returns:
        Lowercase hex digest string (64 characters).

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)  # 64 KB
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


# ── Verification ─────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("hashing.py: running verification\n")

    # 1. Model hashing is deterministic.
    sd = {"b.weight": torch.randn(3, 3), "a.bias": torch.randn(3)}
    h1 = hash_model(sd)
    h2 = hash_model(sd)
    assert h1 == h2, "Same state_dict must produce same hash"
    assert len(h1) == 64, f"Expected 64-char hex, got {len(h1)}"
    print(f"[1/4] Deterministic model hash: {h1[:16]}...")

    # 2. Key order doesn't matter (sorted internally).
    from collections import OrderedDict
    sd_reversed = OrderedDict(reversed(list(sd.items())))
    h3 = hash_model(sd_reversed)
    assert h3 == h1, "Hash must be independent of key insertion order"
    print("[2/4] Key-order independence verified")

    # 3. Different weights produce different hashes.
    sd2 = {"a.bias": torch.randn(3), "b.weight": torch.randn(3, 3)}
    h4 = hash_model(sd2)
    assert h4 != h1, "Different weights must produce different hashes"
    print("[3/4] Different weights → different hash")

    # 4. File hashing works.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(b"test content for hashing")
        tmp = f.name
    fh = hash_file(tmp)
    assert len(fh) == 64
    fh2 = hash_file(tmp)
    assert fh == fh2, "Same file must produce same hash"
    print(f"[4/4] File hash: {fh[:16]}...")

    print("\nhashing.py: all checks passed ✓")
