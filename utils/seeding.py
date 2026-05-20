"""
utils/seeding.py — Deterministic Seed Management
=================================================

Central engine for all randomness in the project. Every source of
stochasticity (data partitioning, client selection, model init, local
training order) is derived from a single root seed via purpose-tagged
derivation so that:

  1. A full run is exactly reproducible given the same root seed.
  2. Changing one subsystem's seed (e.g. gold-retrain trial) does NOT
     alter the seed streams of unrelated subsystems.

Usage:
    from utils.seeding import set_global_seed, derive_seed

    set_global_seed(42)                          # lock everything
    partition_seed = derive_seed(42, "partition") # deterministic child
"""

import hashlib
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Pin every PRNG the training pipeline touches.

    Covers Python stdlib, NumPy, PyTorch CPU, and PyTorch CUDA.
    Also sets the two PyTorch flags needed for fully deterministic
    cuDNN behaviour (at a small performance cost).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic convolution algorithms — slower but reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derive_seed(root_seed: int, purpose: str) -> int:
    """Produce a child seed that is deterministic given (root, purpose)
    but statistically independent of seeds derived with a different
    purpose string.

    Uses SHA-256 to map (root_seed, purpose) → 32-bit integer.
    This avoids the pitfall of simple arithmetic derivation (e.g.
    root + offset) where nearby seeds produce correlated PRNG streams.

    Args:
        root_seed: The run-level root seed from config.
        purpose:   A human-readable tag, e.g. "partition", "client_selection",
                   "gold_retrain_3".  Must be unique per use site.

    Returns:
        An integer in [0, 2^31 - 1] suitable for seeding any PRNG.
    """
    payload = f"{root_seed}:{purpose}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    # Take the first 8 hex chars → 32-bit integer, mask to 31-bit positive.
    return int(digest[:8], 16) % (2**31)


def seed_worker(_worker_id: int) -> None:
    """DataLoader worker initializer for reproducible data loading.

    PyTorch DataLoader workers each get their own PRNG state. Without
    explicit seeding, shuffling order varies across runs. Pass this
    function as `worker_init_fn` when constructing DataLoaders.
    """
    worker_seed = torch.initial_seed() % (2**31)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Verify derivation is deterministic and purpose-sensitive.
    s1 = derive_seed(42, "partition")
    s2 = derive_seed(42, "partition")
    s3 = derive_seed(42, "client_selection")
    assert s1 == s2, "Same inputs must produce same seed"
    assert s1 != s3, "Different purposes must produce different seeds"

    # Verify global seeding makes torch ops reproducible.
    set_global_seed(7)
    a = torch.randn(5)
    set_global_seed(7)
    b = torch.randn(5)
    assert torch.equal(a, b), "Global seeding must make torch deterministic"

    print("seeding.py: all checks passed")
    print(f"  derive_seed(42, 'partition')        = {s1}")
    print(f"  derive_seed(42, 'client_selection')  = {s3}")
