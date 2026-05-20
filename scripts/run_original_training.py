"""
scripts/run_original_training.py — Goal 1: Complete FedAvg Training Run
=========================================================================

Trains the original federated model to convergence (~85-90% test accuracy
on CIFAR-10, 200 rounds of FedAvg with 50 non-IID clients).

Produces the evidence bundle at outputs/run_001/:
    manifest.json, config.yaml, participation_log.json,
    final_model.pt, checkpoints/round_010.pt ... round_200.pt

Run with --verify to confirm deterministic reproducibility: trains a
second time with the same seed and asserts bit-identical models.

Usage:
    python scripts/run_original_training.py
    python scripts/run_original_training.py --verify
    python scripts/run_original_training.py --config config/default.yaml

Phase 2, Goal 1 of the dissertation execution roadmap (Section 9.2).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Ensure repo root is on the path when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config
from data.partitioner import partition_cifar10
from evidence.hashing import hash_model
from evidence.participation_log import ParticipationLog
from federation.trainer import train
from utils.seeding import derive_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the original training run."""
    parser = argparse.ArgumentParser(
        description="Goal 1: Complete FedAvg training run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the project config YAML (default: config/default.yaml).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="run_001",
        help="Output directory name for the evidence bundle (default: run_001).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a second time and verify bit-identical reproducibility.",
    )
    return parser.parse_args()


def print_training_summary(run_dir: Path) -> None:
    """Print a summary of the completed training run."""
    log = ParticipationLog.load(run_dir / "participation_log.json")
    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    rounds = log.rounds
    final_round = rounds[-1]

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Evidence bundle:    {run_dir}")
    print(f"  Rounds completed:   {len(rounds)}")
    print(f"  Final accuracy:     {final_round['test_accuracy']:.4f}")
    print(f"  Final loss:         {final_round['test_loss']:.4f}")
    print(f"  Files in manifest:  {len(manifest['file_hashes'])}")
    print(f"  Hash chain valid:   {log.verify_hash_chain()}")

    # Accuracy progression at checkpoint intervals.
    print("\n  Accuracy progression:")
    for entry in rounds:
        r = entry["round_id"]
        if (r + 1) % 10 == 0 or r == 0 or r == len(rounds) - 1:
            print(f"    Round {r + 1:>3d}: {entry['test_accuracy']:.4f}")

    print("=" * 60)


def verify_reproducibility(
    config_path: str,
    run_id: str,
    original_run_dir: Path,
) -> bool:
    """Run training a second time and verify bit-identical results.

    Args:
        config_path:      Path to the config file.
        run_id:           Original run ID (verification run uses _verify suffix).
        original_run_dir: Path to the original run's evidence bundle.

    Returns:
        True if verification passes, False otherwise.
    """
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 60)
    print("  Running identical training with same seed...")

    cfg = load_config(config_path)
    root_seed = cfg.reproducibility.root_seed
    partition_seed = derive_seed(root_seed, "partition")

    partition = partition_cifar10(
        num_clients=cfg.data.num_clients,
        alpha=cfg.data.alpha,
        seed=partition_seed,
        data_root=cfg.data.data_root,
    )

    verify_id = f"{run_id}_verify"
    verify_dir = train(
        config=cfg,
        partition=partition,
        run_seed=root_seed,
        run_id=verify_id,
    )

    # Compare final models.
    print("\n  Comparing final models...")
    original_sd = torch.load(
        original_run_dir / "final_model.pt", weights_only=True,
    )
    verify_sd = torch.load(
        verify_dir / "final_model.pt", weights_only=True,
    )

    all_match = True
    for key in original_sd:
        if not torch.equal(original_sd[key], verify_sd[key]):
            print(f"  MISMATCH: {key}")
            all_match = False

    # Compare per-round hashes.
    original_log = ParticipationLog.load(
        original_run_dir / "participation_log.json",
    )
    verify_log = ParticipationLog.load(
        verify_dir / "participation_log.json",
    )

    for orig, ver in zip(original_log.rounds, verify_log.rounds):
        if orig["global_model_hash_post"] != ver["global_model_hash_post"]:
            print(f"  HASH MISMATCH at round {orig['round_id']}")
            all_match = False

    if all_match:
        original_hash = hash_model(original_sd)
        print(f"\n  PASS: Bit-identical across all {len(original_log)} rounds")
        print(f"  Final model hash: {original_hash}")
    else:
        print("\n  FAIL: Models diverged — reproducibility broken")

    print("=" * 60)
    return all_match


def main() -> None:
    """Run Goal 1: complete FedAvg training with optional reproducibility check."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = str(REPO_ROOT / args.config)
    cfg = load_config(config_path)
    root_seed = cfg.reproducibility.root_seed

    # ── Partition data ───────────────────────────────────────────
    partition_seed = derive_seed(root_seed, "partition")
    logging.info("Partitioning CIFAR-10: %d clients, α=%.1f, seed=%d",
                 cfg.data.num_clients, cfg.data.alpha, partition_seed)

    partition = partition_cifar10(
        num_clients=cfg.data.num_clients,
        alpha=cfg.data.alpha,
        seed=partition_seed,
        data_root=cfg.data.data_root,
    )

    client_sizes = [len(v) for v in partition.values()]
    logging.info("Partition complete: %d clients, samples range [%d, %d], total %d",
                 len(partition), min(client_sizes), max(client_sizes), sum(client_sizes))

    # ── Train ────────────────────────────────────────────────────
    run_dir = train(
        config=cfg,
        partition=partition,
        run_seed=root_seed,
        run_id=args.run_id,
    )

    print_training_summary(run_dir)

    # ── Verify reproducibility (optional) ────────────────────────
    if args.verify:
        passed = verify_reproducibility(config_path, args.run_id, run_dir)
        if not passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
