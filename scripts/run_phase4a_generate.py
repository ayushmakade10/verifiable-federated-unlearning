"""
scripts/run_phase4a_generate.py — Phase 4a: Failure Bundle Generation
========================================================================

Generates evidence bundles for the 17 failure-case experiments defined
in Section 9.2.  Each bundle is a complete evidence directory that the
existing ``run_verification.py`` can consume directly.

Two categories of experiments:

  File-manipulation (Experiments 5, 6): seconds, no GPU.
  Training-based   (Experiments 2, 3, 4): 30–90 min each, GPU required.

Experiments can be generated selectively::

    # Generate everything (several hours on GPU)
    python scripts/run_phase4a_generate.py --all

    # File-manipulation only (seconds, no GPU)
    python scripts/run_phase4a_generate.py --experiments 5 6

    # Training experiments only
    python scripts/run_phase4a_generate.py --experiments 2 3 4

    # Single experiment
    python scripts/run_phase4a_generate.py --experiments 2

    # Skip already-generated bundles
    python scripts/run_phase4a_generate.py --all --skip-existing

Phase 4a of the dissertation execution roadmap (Section 9.2).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torchvision  # pylint: disable=import-error

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config
from data.partitioner import partition_cifar10
from experiments.failure_cases import (
    find_wrong_clients,
    generate_checkpoint_delete,
    generate_checkpoint_swap,
    generate_finetune_masquerade,
    generate_hash_break,
    generate_log_edit,
    generate_manifest_alter,
    generate_model_swap,
    generate_partial_retraining,
    generate_rollback,
    generate_wrong_client_deletion,
)
from utils.seeding import derive_seed

logger = logging.getLogger(__name__)


# ── Experiment Registry ─────────────────────────────────────────


EXPERIMENT_IDS = {
    2: "Partial Retraining (3 configs: K=50, K=100, K=150)",
    3: "Fine-Tuning Masquerade (2 configs: aggressive, subtle)",
    4: "Wrong Client Deletion (2 configs: most different, most similar)",
    5: "Rollback-Only (4 configs: R050/R150 × updated/stale manifest)",
    6: "Inconsistent Evidence (6 configs: 6a–6f)",
}


# ── CLI ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4a: Generate failure-case evidence bundles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Experiment IDs:\n"
            "  2  Partial Retraining (GPU, ~2h)\n"
            "  3  Fine-Tuning Masquerade (GPU, ~30min)\n"
            "  4  Wrong Client Deletion (GPU, ~3h)\n"
            "  5  Rollback-Only (no GPU, seconds)\n"
            "  6  Inconsistent Evidence (no GPU, seconds)\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true",
        help="Generate all 17 failure bundles.",
    )
    group.add_argument(
        "--experiments", type=int, nargs="+",
        choices=[2, 3, 4, 5, 6],
        help="Generate specific experiments by ID.",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to the project config YAML.",
    )
    parser.add_argument(
        "--target-client", type=int, default=0,
        help="Target client for unlearning (default: 0).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip experiments whose output directory already exists.",
    )
    return parser.parse_args()


# ── Orchestration ───────────────────────────────────────────────


def _should_skip(run_id: str, output_base: Path, skip_existing: bool) -> bool:
    """Check whether an experiment bundle already exists."""
    dest = output_base / run_id
    if skip_existing and dest.exists():
        logger.info("SKIP %s (already exists)", run_id)
        return True
    return False


def run_experiment_2(
    cfg, partition_minus_target, root_seed, output_base,
    original_bundle, skip_existing,
):
    """Generate Experiment 2: Partial Retraining (3 configs)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Partial Retraining")
    logger.info("=" * 60)

    original_log_path = original_bundle / "participation_log.json"

    for resume_round in [50, 100, 150]:
        run_id = f"phase4a/exp2_partial_K{resume_round:03d}"
        if _should_skip(run_id, output_base, skip_existing):
            continue

        ckpt_path = (
            original_bundle / "checkpoints" / f"round_{resume_round:03d}.pt"
        )
        if not ckpt_path.exists():
            logger.error(
                "  Checkpoint not found: %s — skipping K=%d",
                ckpt_path, resume_round,
            )
            continue

        start = time.time()
        generate_partial_retraining(
            config=cfg,
            partition_minus_target=partition_minus_target,
            run_seed=root_seed,
            resume_round=resume_round,
            checkpoint_path=ckpt_path,
            original_log_path=original_log_path,
            run_id=run_id,
            source_bundle=original_bundle,
        )
        elapsed = time.time() - start
        logger.info("  Completed K=%d in %.1f seconds", resume_round, elapsed)


def run_experiment_3(
    cfg, partition_minus_target, root_seed, output_base,
    original_bundle, skip_existing,
):
    """Generate Experiment 3: Fine-Tuning Masquerade (2 configs)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Fine-Tuning Masquerade")
    logger.info("=" * 60)

    original_model_path = original_bundle / "final_model.pt"

    variants = [
        ("aggressive", 20, 0.01),
        ("subtle", 10, 0.001),
    ]

    for name, num_rounds, lr in variants:
        run_id = f"phase4a/exp3_finetune_{name}"
        if _should_skip(run_id, output_base, skip_existing):
            continue

        finetune_seed = derive_seed(root_seed, f"finetune_{name}")

        start = time.time()
        generate_finetune_masquerade(
            config=cfg,
            partition_minus_target=partition_minus_target,
            run_seed=finetune_seed,
            original_model_path=original_model_path,
            num_rounds=num_rounds,
            learning_rate=lr,
            run_id=run_id,
            source_bundle=original_bundle,
        )
        elapsed = time.time() - start
        logger.info(
            "  Completed %s (%d rounds, lr=%.4f) in %.1f seconds",
            name, num_rounds, lr, elapsed,
        )


def run_experiment_4(
    cfg, full_partition, root_seed, output_base,
    original_bundle, target_client, skip_existing,
):
    """Generate Experiment 4: Wrong Client Deletion (2 configs)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Wrong Client Deletion")
    logger.info("=" * 60)

    # Load dataset labels for histogram computation.
    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.data_root, train=True, download=True,
    )
    labels = np.array(train_dataset.targets)

    # Identify wrong clients.
    wrong_clients = find_wrong_clients(
        full_partition, target_client, labels, cfg.model.num_classes,
    )

    # Save wrong-client identification for the dissertation.
    report_dir = Path(cfg.checkpoint.output_dir) / "phase4a" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "wrong_client_identification.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(wrong_clients, f, indent=2)
    logger.info("  Wrong-client report saved to %s", report_path)

    variants = [
        ("different", wrong_clients["most_different"]["client_id"]),
        ("similar", wrong_clients["most_similar"]["client_id"]),
    ]

    for label, wrong_id in variants:
        run_id = f"phase4a/exp4_wrong_{label}"
        if _should_skip(run_id, output_base, skip_existing):
            continue

        wrong_seed = derive_seed(root_seed, f"wrong_client_{wrong_id}")

        start = time.time()
        generate_wrong_client_deletion(
            config=cfg,
            full_partition=full_partition,
            run_seed=wrong_seed,
            wrong_client_id=wrong_id,
            run_id=run_id,
            source_bundle=original_bundle,
        )
        elapsed = time.time() - start
        logger.info(
            "  Completed %s (client %d) in %.1f seconds",
            label, wrong_id, elapsed,
        )


def run_experiment_5(
    output_base, original_bundle, skip_existing,
):
    """Generate Experiment 5: Rollback-Only (4 configs)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Rollback-Only")
    logger.info("=" * 60)

    for checkpoint_round in [50, 150]:
        for update_manifest in [True, False]:
            suffix = "updated" if update_manifest else "stale"
            run_id = f"phase4a/exp5_rollback_R{checkpoint_round:03d}_{suffix}"
            if _should_skip(run_id, output_base, skip_existing):
                continue

            start = time.time()
            generate_rollback(
                source_bundle=original_bundle,
                checkpoint_round=checkpoint_round,
                update_manifest=update_manifest,
                run_id=run_id,
                output_base=output_base,
            )
            elapsed = time.time() - start
            logger.info(
                "  Completed R=%d %s in %.1f seconds",
                checkpoint_round, suffix, elapsed,
            )


def run_experiment_6(
    output_base, original_bundle, gold_base, skip_existing,
):
    """Generate Experiment 6: Inconsistent Evidence (6 configs)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: Inconsistent Evidence")
    logger.info("=" * 60)

    # Locate gold trial models/checkpoints for swap operations.
    gold_trial_0 = gold_base / "trial_00"
    gold_trial_1 = gold_base / "trial_01"
    gold_model_path = gold_trial_0 / "final_model.pt"
    gold_ckpt_path = gold_trial_1 / "checkpoints" / "round_100.pt"

    # Validate required gold artifacts exist.
    for path, desc in [
        (gold_model_path, "gold trial 0 final model"),
        (gold_ckpt_path, "gold trial 1 round-100 checkpoint"),
    ]:
        if not path.exists():
            logger.error("  Required file missing: %s (%s)", path, desc)
            logger.error("  Cannot generate Experiment 6 — aborting")
            return

    # 6a: Model swap (gold model into original bundle, keep manifest).
    run_id = "phase4a/exp6a_model_swap"
    if not _should_skip(run_id, output_base, skip_existing):
        generate_model_swap(
            source_bundle=original_bundle,
            swap_model_path=gold_model_path,
            run_id=run_id,
            output_base=output_base,
        )

    # 6b: Log edit (modify round 100's selected_clients).
    run_id = "phase4a/exp6b_log_edit"
    if not _should_skip(run_id, output_base, skip_existing):
        generate_log_edit(
            source_bundle=original_bundle,
            target_round=100,
            run_id=run_id,
            output_base=output_base,
        )

    # 6c: Hash chain break (corrupt round 100's post-hash).
    run_id = "phase4a/exp6c_hash_break"
    if not _should_skip(run_id, output_base, skip_existing):
        generate_hash_break(
            source_bundle=original_bundle,
            target_round=100,
            run_id=run_id,
            output_base=output_base,
        )

    # 6d: Checkpoint swap (replace round 100 checkpoint with gold trial 1's).
    run_id = "phase4a/exp6d_checkpoint_swap"
    if not _should_skip(run_id, output_base, skip_existing):
        generate_checkpoint_swap(
            source_bundle=original_bundle,
            target_round=100,
            swap_checkpoint_path=gold_ckpt_path,
            run_id=run_id,
            output_base=output_base,
        )

    # 6e: Checkpoint deletion (delete round 100 checkpoint).
    run_id = "phase4a/exp6e_checkpoint_delete"
    if not _should_skip(run_id, output_base, skip_existing):
        generate_checkpoint_delete(
            source_bundle=original_bundle,
            target_round=100,
            run_id=run_id,
            output_base=output_base,
        )

    # 6f: Manifest alter (swap model + fix manifest hash).
    run_id = "phase4a/exp6f_manifest_alter"
    if not _should_skip(run_id, output_base, skip_existing):
        generate_manifest_alter(
            source_bundle=original_bundle,
            swap_model_path=gold_model_path,
            run_id=run_id,
            output_base=output_base,
        )


# ── Main ────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for Phase 4a bundle generation."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = str(REPO_ROOT / args.config)
    cfg = load_config(config_path)
    root_seed = cfg.reproducibility.root_seed
    target_client = args.target_client
    output_base = Path(cfg.checkpoint.output_dir)

    # Locate the original training run.
    original_bundle = output_base / "run_001"
    if not original_bundle.exists():
        logger.error(
            "Original training bundle not found at %s — "
            "run run_original_training.py first",
            original_bundle,
        )
        sys.exit(1)

    # Gold models base path.
    gold_base = output_base / "gold" / f"client_{target_client}"

    # Reconstruct full partition (needed for Experiments 2, 3, 4).
    partition_seed = derive_seed(root_seed, "partition")
    full_partition = partition_cifar10(
        num_clients=cfg.data.num_clients,
        alpha=cfg.data.alpha,
        seed=partition_seed,
        data_root=cfg.data.data_root,
    )

    # Partition with target client removed.
    partition_minus_target = {
        k: v for k, v in full_partition.items() if k != target_client
    }

    # Determine which experiments to run.
    experiments = (
        sorted(EXPERIMENT_IDS.keys()) if args.all
        else sorted(args.experiments)
    )

    logger.info("Phase 4a Generation — experiments: %s", experiments)
    logger.info("Output base: %s", output_base)
    logger.info("Original bundle: %s", original_bundle)
    logger.info("Target client: %d", target_client)

    overall_start = time.time()

    for exp_id in experiments:
        logger.info("\n%s", EXPERIMENT_IDS[exp_id])

        if exp_id == 2:
            run_experiment_2(
                cfg, partition_minus_target, root_seed, output_base,
                original_bundle, args.skip_existing,
            )
        elif exp_id == 3:
            run_experiment_3(
                cfg, partition_minus_target, root_seed, output_base,
                original_bundle, args.skip_existing,
            )
        elif exp_id == 4:
            run_experiment_4(
                cfg, full_partition, root_seed, output_base,
                original_bundle, target_client, args.skip_existing,
            )
        elif exp_id == 5:
            run_experiment_5(
                output_base, original_bundle, args.skip_existing,
            )
        elif exp_id == 6:
            run_experiment_6(
                output_base, original_bundle, gold_base,
                args.skip_existing,
            )

    overall_elapsed = time.time() - overall_start
    logger.info("%s", "=" * 60)
    logger.info(
        "Phase 4a generation complete in %.1f seconds", overall_elapsed,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
