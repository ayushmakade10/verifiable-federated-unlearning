"""
scripts/run_gold_retraining.py — Goal 2: Gold-Standard Retraining
===================================================================

Runs 10 independent retraining trials excluding a target client.
Each trial uses a fresh independent seed derived via
    derive_seed(root_seed, f"gold_retrain_{trial}")
and resamples clients from the 49 remaining clients (full resample,
not preserved schedule).

Storage layout (Section 6.5):
    outputs/gold/client_{k}/
    ├── trial_00/   (full checkpoints)
    ├── trial_01/   (full checkpoints)
    ├── trial_02/   (full checkpoints)
    ├── trial_03/   (final model only)
    ├── ...
    └── trial_09/   (final model only)

Usage:
    python scripts/run_gold_retraining.py
    python scripts/run_gold_retraining.py --target-client 0
    python scripts/run_gold_retraining.py --target-client 7 --trials 0 1 2
    python scripts/run_gold_retraining.py --full-checkpoint-trials 3

Phase 2, Goal 2 of the dissertation execution roadmap (Section 9.2).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config
from data.partitioner import partition_cifar10
from evidence.hashing import hash_model
from evidence.participation_log import ParticipationLog
from federation.trainer import train
from utils.seeding import derive_seed


# Number of trials that get full checkpoint storage (Section 2.2).
DEFAULT_FULL_CHECKPOINT_TRIALS = 3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for gold-standard retraining."""
    parser = argparse.ArgumentParser(
        description="Goal 2: Gold-standard retraining pipeline.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the project config YAML.",
    )
    parser.add_argument(
        "--target-client",
        type=int,
        default=0,
        help="Client ID to exclude from retraining (default: 0).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        nargs="*",
        default=None,
        help="Specific trial indices to run (default: all 10). "
             "Use to resume after a crash, e.g. --trials 5 6 7 8 9.",
    )
    parser.add_argument(
        "--full-checkpoint-trials",
        type=int,
        default=DEFAULT_FULL_CHECKPOINT_TRIALS,
        help="Number of leading trials that save full checkpoints "
             f"(default: {DEFAULT_FULL_CHECKPOINT_TRIALS}).",
    )
    return parser.parse_args()


def remove_client(
    partition: dict[int, list[int]],
    client_id: int,
) -> dict[int, list[int]]:
    """Return a copy of the partition with the target client removed.

    Args:
        partition:  Full client-to-indices mapping.
        client_id:  Client to exclude.

    Returns:
        New partition dict without the target client.

    Raises:
        KeyError: If client_id is not in the partition.
    """
    if client_id not in partition:
        raise KeyError(
            f"Target client {client_id} not found in partition "
            f"(available: {sorted(partition.keys())})"
        )
    return {k: v for k, v in partition.items() if k != client_id}


def main() -> None:
    """Run Goal 2: gold-standard retraining trials excluding target client."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("gold_retraining")

    config_path = str(REPO_ROOT / args.config)
    cfg = load_config(config_path)
    root_seed = cfg.reproducibility.root_seed
    num_trials = cfg.gold_standard.num_trials
    target_client = args.target_client

    # Determine which trials to run.
    if args.trials is not None:
        trial_indices = args.trials
        for t in trial_indices:
            if t < 0 or t >= num_trials:
                logger.error("Trial index %d out of range [0, %d)", t, num_trials)
                sys.exit(1)
    else:
        trial_indices = list(range(num_trials))

    logger.info(
        "Gold-standard retraining: target_client=%d, trials=%s, "
        "full_checkpoints=%d",
        target_client, trial_indices, args.full_checkpoint_trials,
    )

    # ── Partition data and remove target client ──────────────────
    partition_seed = derive_seed(root_seed, "partition")
    full_partition = partition_cifar10(
        num_clients=cfg.data.num_clients,
        alpha=cfg.data.alpha,
        seed=partition_seed,
        data_root=cfg.data.data_root,
    )

    target_samples = len(full_partition[target_client])
    reduced_partition = remove_client(full_partition, target_client)
    remaining_samples = sum(len(v) for v in reduced_partition.values())

    logger.info(
        "Removed client %d (%d samples). Remaining: %d clients, %d samples.",
        target_client, target_samples, len(reduced_partition), remaining_samples,
    )

    # ── Run trials ───────────────────────────────────────────────
    gold_base_dir = Path(cfg.checkpoint.output_dir) / "gold" / f"client_{target_client}"
    results = []

    for trial in trial_indices:
        # Independent fresh seed per trial (Section 2.2).
        trial_seed = derive_seed(root_seed, f"gold_retrain_{trial}")
        run_id = f"gold/client_{target_client}/trial_{trial:02d}"

        # Full checkpoints for leading trials, final model only for the rest.
        save_ckpts = trial < args.full_checkpoint_trials

        logger.info(
            "━━━ Trial %d/%d ━━━ seed=%d, checkpoints=%s",
            trial, num_trials, trial_seed, save_ckpts,
        )

        run_dir = train(
            config=cfg,
            partition=reduced_partition,
            run_seed=trial_seed,
            run_id=run_id,
            save_checkpoints=save_ckpts,
        )

        # Record result.
        log = ParticipationLog.load(run_dir / "participation_log.json")
        final_round = log.rounds[-1]
        final_sd = torch.load(run_dir / "final_model.pt", weights_only=True)
        final_hash = hash_model(final_sd)

        results.append({
            "trial": trial,
            "seed": trial_seed,
            "final_accuracy": final_round["test_accuracy"],
            "final_loss": final_round["test_loss"],
            "final_model_hash": final_hash,
            "checkpoints_saved": save_ckpts,
            "run_dir": str(run_dir),
        })

        logger.info(
            "Trial %d complete: acc=%.4f, loss=%.4f, hash=%s...",
            trial, final_round["test_accuracy"],
            final_round["test_loss"], final_hash[:16],
        )

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"GOLD-STANDARD RETRAINING COMPLETE — Target Client {target_client}")
    print("=" * 70)
    print(f"  {'Trial':>5}  {'Accuracy':>8}  {'Loss':>8}  {'Ckpts':>5}  {'Hash':>18}")
    print("  " + "-" * 55)

    accuracies = []
    for r in results:
        accuracies.append(r["final_accuracy"])
        ckpt_str = "full" if r["checkpoints_saved"] else "final"
        print(
            f"  {r['trial']:>5d}  {r['final_accuracy']:>8.4f}  "
            f"{r['final_loss']:>8.4f}  {ckpt_str:>5}  {r['final_model_hash'][:18]}"
        )

    if len(accuracies) >= 2:
        acc_range = max(accuracies) - min(accuracies)
        acc_mean = sum(accuracies) / len(accuracies)
        print("  " + "-" * 55)
        print(f"  Mean accuracy:  {acc_mean:.4f}")
        print(f"  Accuracy range: {acc_range:.4f} (expect ~1-2%)")

        if acc_range > 0.05:
            print("  WARNING: Accuracy spread > 5% — check for convergence issues.")

    # Save summary JSON for Goal 3 consumption.
    summary_path = gold_base_dir / "retraining_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_client": target_client,
            "root_seed": root_seed,
            "num_trials": len(results),
            "trials": results,
        }, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
