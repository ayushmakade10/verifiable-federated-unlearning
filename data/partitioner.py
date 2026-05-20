"""
data/partitioner.py — Dirichlet Non-IID Data Partitioning
=========================================================

Splits a dataset across N federated clients using a Dirichlet
distribution over class labels, which is the standard approach for
simulating non-IID federated settings (Hsu et al., 2019).

Key design decisions:
  - Deterministic given (seed, alpha, num_clients).
  - Partition is saved as JSON so the auditor can reconstruct it
    exactly without needing the raw data generation code.
  - Minimum-samples guard prevents degenerate empty clients.

Usage:
    from data.partitioner import partition_cifar10, save_partition

    mapping = partition_cifar10(num_clients=50, alpha=0.3, seed=12345)
    save_partition(mapping, "outputs/run_001/partition.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torchvision
import torchvision.transforms as transforms


def partition_by_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 10,
) -> Dict[int, List[int]]:
    """Assign dataset indices to clients via Dirichlet partitioning.

    For each class, we draw a probability vector from Dir(alpha) over
    the clients and allocate that class's samples proportionally.

    Args:
        labels:     1-D array of integer class labels for the full dataset.
        num_clients: Number of federated clients.
        alpha:      Dirichlet concentration parameter.
                    Lower → more heterogeneous (clients specialise in fewer classes).
        seed:       Random seed for reproducibility.
        min_samples_per_client:
                    If any client receives fewer than this many samples,
                    we redistribute from the largest client. Prevents
                    degenerate single-sample clients that break training.

    Returns:
        Dict mapping client_id (int) → list of dataset indices (List[int]).
    """
    rng = np.random.RandomState(seed)
    num_classes = int(labels.max()) + 1

    # Group sample indices by class.
    class_indices: Dict[int, np.ndarray] = {
        c: np.where(labels == c)[0] for c in range(num_classes)
    }

    # Initialise empty client buckets.
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    # For each class, draw a Dirichlet proportion vector and distribute.
    for c in range(num_classes):
        indices = class_indices[c]
        rng.shuffle(indices)

        # Dir(alpha, ..., alpha) with num_clients dimensions.
        proportions = rng.dirichlet([alpha] * num_clients)

        # Convert proportions to cumulative sample counts.
        counts = (proportions * len(indices)).astype(int)
        # Assign any rounding remainder to a random client.
        remainder = len(indices) - counts.sum()
        if remainder > 0:
            lucky = rng.choice(num_clients, size=remainder, replace=False)
            counts[lucky] += 1

        # Distribute indices according to counts.
        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(indices[start:end].tolist())
            start = end

    # Guard: redistribute if any client is below minimum.
    for client_id in range(num_clients):
        while len(client_indices[client_id]) < min_samples_per_client:
            # Find the client with the most samples.
            donor = max(client_indices, key=lambda k: len(client_indices[k]))
            # Move one sample from donor to this client.
            client_indices[client_id].append(client_indices[donor].pop())

    # Sort each client's indices for deterministic ordering.
    for client_id in client_indices:
        client_indices[client_id].sort()

    return client_indices


def partition_cifar10(
    num_clients: int,
    alpha: float,
    seed: int,
    data_root: str = "./data/raw",
) -> Dict[int, List[int]]:
    """Convenience wrapper: download CIFAR-10 and partition it.

    Returns:
        Dict mapping client_id → list of CIFAR-10 training set indices.
    """
    # Download if necessary (transform is irrelevant; we only need labels).
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    labels = np.array(dataset.targets)
    return partition_by_dirichlet(labels, num_clients, alpha, seed)


# ── Persistence ──────────────────────────────────────────────────────


def save_partition(
    partition: Dict[int, List[int]],
    path: str | Path,
) -> None:
    """Save a partition mapping to JSON.

    Keys are stringified because JSON requires string keys.
    The auditor loads this file to reconstruct the exact same split.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {str(k): v for k, v in partition.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f)


def load_partition(path: str | Path) -> Dict[int, List[int]]:
    """Load a saved partition, restoring integer keys."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── Diagnostics ──────────────────────────────────────────────────────


def partition_summary(
    partition: Dict[int, List[int]],
    labels: np.ndarray,
    num_classes: int = 10,
) -> None:
    """Print a human-readable summary of the partition.

    Shows per-client sample count and class distribution so you
    can visually confirm non-IID behaviour.
    """
    print(f"{'Client':>8} | {'Samples':>7} | Class Distribution")
    print("-" * 60)
    for client_id in sorted(partition.keys()):
        indices = partition[client_id]
        client_labels = labels[indices]
        counts = np.bincount(client_labels, minlength=num_classes)
        dist_str = " ".join(f"{c:4d}" for c in counts)
        print(f"{client_id:>8} | {len(indices):>7} | {dist_str}")

    total = sum(len(v) for v in partition.values())
    print("-" * 60)
    print(f"{'Total':>8} | {total:>7} |")


# ── Verification ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("partitioner.py: running verification\n")

    # Try CIFAR-10; fall back to synthetic labels if download is blocked.
    try:
        print("[1/4] Attempting CIFAR-10 download ...")
        partition = partition_cifar10(num_clients=50, alpha=0.3, seed=42)
        ds = torchvision.datasets.CIFAR10(root="./data/raw", train=True, download=False)
        labels = np.array(ds.targets)
        num_samples = 50000
        using_real = True
        print("  Using real CIFAR-10 labels.")
    except Exception:
        print(
            "  CIFAR-10 unavailable — using synthetic labels (50k samples, 10 classes)."
        )
        rng = np.random.RandomState(0)
        num_samples = 50000
        labels = rng.randint(0, 10, size=num_samples)
        partition = partition_by_dirichlet(labels, num_clients=50, alpha=0.3, seed=42)
        using_real = False

    # 2. Basic integrity checks.
    print("[2/4] Running integrity checks ...")
    all_indices = sorted(idx for idxs in partition.values() for idx in idxs)
    assert (
        len(all_indices) == num_samples
    ), f"Expected {num_samples}, got {len(all_indices)}"
    assert len(set(all_indices)) == num_samples, "Duplicate indices detected"
    assert all(len(v) >= 10 for v in partition.values()), "Client below minimum"
    print(f"  ✓ All {num_samples:,} samples assigned, no duplicates, no empty clients.")

    # 3. Reproducibility check.
    print("[3/4] Verifying reproducibility ...")
    partition2 = partition_by_dirichlet(labels, num_clients=50, alpha=0.3, seed=42)
    assert partition == partition2, "Same seed must produce same partition"
    partition3 = partition_by_dirichlet(labels, num_clients=50, alpha=0.3, seed=99)
    assert partition != partition3, "Different seed must produce different partition"
    print(
        "  ✓ Deterministic: same seed → same split, different seed → different split."
    )

    # 4. Save/load round-trip.
    print("[4/4] Verifying save/load round-trip ...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    save_partition(partition, tmp_path)
    loaded = load_partition(tmp_path)
    assert loaded == partition, "Save/load must be lossless"
    print(f"  ✓ Round-trip via {tmp_path} successful.")

    # Show first 10 clients for visual inspection.
    print("\n── Distribution Preview (first 10 clients) ──")
    preview = {k: v for k, v in partition.items() if k < 10}
    partition_summary(preview, labels)

    print("\npartitioner.py: all checks passed ✓")
    if not using_real:
        print("  (Run locally with network access to verify against real CIFAR-10.)")
