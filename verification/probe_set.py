"""
verification/probe_set.py — Privacy-Preserving Probe Set Construction
=======================================================================

Constructs a class-weighted evaluation subset from the server's held-out
test set, using the target client's class histogram to maximise
discriminative power.

The auditor probes exactly where the deleted client's influence is
strongest — classes the client specialised in — without ever accessing
the client's actual training data.

Privacy properties (Section 4.6):
  - No target client raw data used — only test/evaluation images
  - Only the target client's class histogram (10 numbers, voluntarily
    provided in the unlearning request)
  - Works with numeric labels only — no semantic class knowledge

Usage:
    from verification.probe_set import build_probe_set

    probe_loader, full_loader = build_probe_set(
        test_dataset=cifar10_test,
        client_class_histogram=[30, 45, 20, 300, 15, 25, 10, 150, 35, 40],
        batch_size=128,
    )
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


def build_probe_set(
    test_dataset: torch.utils.data.Dataset,
    client_class_histogram: List[int],
    batch_size: int = 128,
    probe_size: int = 2000,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build the class-weighted probe set and a full test set DataLoader.

    The probe set over-samples test images from classes the target client
    specialised in (had the most training data for), making checks more
    sensitive to the effect of removing that client's data.

    Args:
        test_dataset: The full test/evaluation dataset (e.g. CIFAR-10 test).
            Must support indexing and return (image, label) pairs where
            labels are integer class indices.
        client_class_histogram: Number of training samples the target
            client had per class. Length must equal the number of classes.
            Used only for weighting — larger counts → more probe samples
            from that class.
        batch_size: Batch size for both DataLoaders.
        probe_size: Number of samples to draw for the probe set.
            Default 2000 (~20% of CIFAR-10 test set).
        num_workers: DataLoader worker count.

    Returns:
        Tuple of (probe_loader, full_loader):
          - probe_loader: Class-weighted subset DataLoader (primary eval)
          - full_loader: Full test set DataLoader (secondary sanity check)
    """
    histogram = torch.tensor(client_class_histogram, dtype=torch.float32)
    num_classes = len(client_class_histogram)

    # Compute per-class weight proportional to client's data distribution.
    # Classes with more client data get higher weight → more probe samples.
    total_samples = histogram.sum()
    if total_samples > 0:
        class_weights = histogram / total_samples
    else:
        # Fallback: uniform if histogram is all zeros.
        class_weights = torch.ones(num_classes) / num_classes

    # Assign per-sample weight based on its class label.
    targets = _extract_targets(test_dataset)
    sample_weights = torch.zeros(len(targets))
    for idx, label in enumerate(targets):
        if 0 <= label < num_classes:
            sample_weights[idx] = class_weights[label].item()

    # Weighted random sampler draws probe_size samples.
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=min(probe_size, len(targets)),
        replacement=True,
    )

    probe_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    full_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return probe_loader, full_loader


def build_deterministic_probe_set(
    test_dataset: torch.utils.data.Dataset,
    client_class_histogram: List[int],
    batch_size: int = 128,
    probe_size: int = 2000,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build a deterministic (non-random) probe set for reproducible evaluation.

    Instead of weighted random sampling, selects a fixed subset of test
    images proportional to the client's class histogram. This ensures
    identical probe sets across multiple verification runs.

    Preferred for calibration and final verdict. The random variant is
    available for sensitivity analysis.

    Args:
        test_dataset: The full test/evaluation dataset.
        client_class_histogram: Per-class sample counts for the target client.
        batch_size: Batch size for both DataLoaders.
        probe_size: Total number of probe samples to select.
        num_workers: DataLoader worker count.

    Returns:
        Tuple of (probe_loader, full_loader).
    """
    histogram = torch.tensor(client_class_histogram, dtype=torch.float32)
    num_classes = len(client_class_histogram)
    total_client_samples = histogram.sum().item()

    # Per-class quota proportional to histogram.
    if total_client_samples > 0:
        class_fractions = histogram / total_client_samples
    else:
        class_fractions = torch.ones(num_classes) / num_classes

    class_quotas = (class_fractions * probe_size).int().tolist()

    # Ensure we hit the target size (rounding may lose a few).
    shortfall = probe_size - sum(class_quotas)
    # Distribute shortfall to classes with largest fractional parts.
    fractional_parts = [
        (class_fractions[c].item() * probe_size) - class_quotas[c]
        for c in range(num_classes)
    ]
    for _ in range(max(0, shortfall)):
        best_class = int(
            max(range(num_classes), key=lambda c: fractional_parts[c])
        )
        class_quotas[best_class] += 1
        fractional_parts[best_class] -= 1.0

    # Group test indices by class.
    targets = _extract_targets(test_dataset)
    class_indices: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(targets):
        if 0 <= label < num_classes:
            class_indices[label].append(idx)

    # Select first N indices from each class (deterministic).
    selected_indices = []
    for cls in range(num_classes):
        available = class_indices[cls]
        quota = min(class_quotas[cls], len(available))
        selected_indices.extend(available[:quota])

    probe_subset = Subset(test_dataset, selected_indices)
    probe_loader = DataLoader(
        probe_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    full_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return probe_loader, full_loader


def _extract_targets(dataset: torch.utils.data.Dataset) -> List[int]:
    """Extract integer class labels from a dataset.

    Handles both datasets with a `.targets` attribute (e.g. CIFAR-10)
    and generic datasets that require indexing.

    Args:
        dataset: The dataset to extract labels from.

    Returns:
        List of integer class labels.
    """
    if hasattr(dataset, "targets"):
        return list(dataset.targets)

    # Fallback: iterate and extract labels.
    labels = []
    for item in dataset:
        _, label = item
        labels.append(int(label))
    return labels
