"""
scripts/create_unlearning_request.py — Generate unlearning_request.json
=========================================================================

Creates the unlearning request file for a target client, including
the client_class_histogram field needed by the probe set (Section 4.6).

Reads the partition to compute how many training samples the target
client had per class, then writes the request into the specified
evidence bundle directory.

Usage:
    python scripts/create_unlearning_request.py --target-client 0 --bundle-dir outputs/run_001
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torchvision  # pylint: disable=import-error

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config
from data.partitioner import partition_by_dirichlet
from utils.seeding import derive_seed


def main() -> None:
    """Generate unlearning_request.json with client class histogram."""
    parser = argparse.ArgumentParser(
        description="Create unlearning_request.json for a target client.",
    )
    parser.add_argument(
        "--target-client", type=int, default=0,
        help="Client ID requesting deletion (default: 0).",
    )
    parser.add_argument(
        "--bundle-dir", type=str, required=True,
        help="Evidence bundle directory to write the request into.",
    )
    parser.add_argument(
        "--source-run", type=str, default="run_001",
        help="Source run ID (default: run_001).",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to project config YAML.",
    )
    args = parser.parse_args()

    cfg = load_config(str(REPO_ROOT / args.config))

    # Load training dataset labels.
    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.data_root, train=True, download=True,
    )
    labels = np.array(train_dataset.targets)

    # Reconstruct partition.
    partition_seed = derive_seed(cfg.reproducibility.root_seed, "partition")
    partition = partition_by_dirichlet(
        labels=labels,
        num_clients=cfg.data.num_clients,
        alpha=cfg.data.alpha,
        seed=partition_seed,
    )

    # Compute class histogram for the target client.
    client_indices = partition[args.target_client]
    client_labels = labels[client_indices]
    counter = Counter(int(lbl) for lbl in client_labels)
    histogram = [counter.get(c, 0) for c in range(cfg.model.num_classes)]

    # Build request object (Section 3.3).
    request = {
        "request_id": f"req_{args.target_client:03d}",
        "target_client_id": args.target_client,
        "source_run_id": args.source_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_type": "client_deletion",
        "client_class_histogram": histogram,
    }

    bundle_dir = Path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    request_path = bundle_dir / "unlearning_request.json"
    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)

    print(f"Unlearning request created: {request_path}")
    print(f"  Target client: {args.target_client}")
    print(f"  Total samples: {sum(histogram)}")
    print(f"  Class histogram: {histogram}")
    print("  Dominant classes: ", end="")
    sorted_classes = sorted(range(len(histogram)), key=lambda c: histogram[c], reverse=True)
    for cls in sorted_classes[:3]:
        print(f"class {cls} ({histogram[cls]} samples) ", end="")
    print()


if __name__ == "__main__":
    main()
