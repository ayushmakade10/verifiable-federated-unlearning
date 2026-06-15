"""
scripts/run_phase4b_federaser_prep.py — FedEraser Prep Training
=================================================================

Runs the pre-requisite re-training with per-client update storage,
into an ISOLATED directory (outputs/run_001_federaser_prep/),
completely separate from outputs/run_001/.

A single Δt=5 prep run stores updates at rounds 5, 10, 15, ..., 200.
This is a superset of the rounds needed by all three FedEraser configs
(Δt=10, Δt=20, Δt=5), so one prep run serves all of them.

CRITICAL: After training, the final model hash MUST match the original
(b785611987...). If it doesn't, the storage hook introduced a side
effect — the run aborts with an error.

Usage::

    python scripts/run_phase4b_federaser_prep.py

    # Custom storage interval (default 5):
    python scripts/run_phase4b_federaser_prep.py --delta-t 5

Phase 4b of the dissertation execution roadmap (Section 9.3, Method 4).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position
from config.schemas import load_config
from data.partitioner import partition_cifar10
from evidence.participation_log import ParticipationLog
from federation.trainer_federaser import train_with_update_storage
from utils.seeding import derive_seed

logger = logging.getLogger(__name__)

# The original training's final model hash (Phase 2). The prep run must
# reproduce this bit-for-bit.
ORIGINAL_FINAL_HASH = (
    "b785611987cc13b06e622a414b18d2c044c49f3dcd12f20353115385779267ea"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FedEraser prep: re-train with per-client update storage.",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to the project config YAML.",
    )
    parser.add_argument(
        "--delta-t", type=int, default=5,
        help="Storage interval (default 5 — serves all configs).",
    )
    parser.add_argument(
        "--run-id", type=str, default="run_001_federaser_prep",
        help="Output directory name (isolated from run_001).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip if the prep run already has a final_model.pt.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the FedEraser prep training."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = str(REPO_ROOT / args.config)
    cfg = load_config(config_path)
    output_base = Path(cfg.checkpoint.output_dir)
    run_dir = output_base / args.run_id

    # ── Safety: never touch outputs/run_001 ──────────────────────
    if args.run_id == "run_001":
        logger.error(
            "Refusing to use run_id='run_001' — that would overwrite "
            "the original training. Use a separate directory.",
        )
        sys.exit(1)

    if args.skip_existing and (run_dir / "final_model.pt").exists():
        logger.info("Prep run already exists at %s — skipping.", run_dir)
        sys.exit(0)

    # ── Read original run_seed from run_001's participation log ──
    original_log_path = output_base / "run_001" / "participation_log.json"
    if not original_log_path.exists():
        logger.error(
            "Original participation log not found at %s", original_log_path,
        )
        sys.exit(1)

    original_log = ParticipationLog.load(str(original_log_path))
    original_seed = original_log.run_seed
    logger.info(
        "Read original run_seed=%d from %s", original_seed, original_log_path,
    )

    # ── Reconstruct the partition (same seed as original) ────────
    partition_seed = derive_seed(cfg.reproducibility.root_seed, "partition")
    partition = partition_cifar10(
        num_clients=cfg.data.num_clients,
        alpha=cfg.data.alpha,
        seed=partition_seed,
    )

    logger.info(
        "Starting FedEraser prep: Δt=%d, storing at rounds %d,%d,...,%d",
        args.delta_t, args.delta_t, 2 * args.delta_t,
        cfg.federation.num_rounds,
    )
    logger.info(
        "This reproduces the original 200-round training EXACTLY, "
        "plus update storage. Expected to match hash %s",
        ORIGINAL_FINAL_HASH[:16] + "...",
    )

    start = time.time()
    train_with_update_storage(
        config=cfg,
        partition=partition,
        run_seed=original_seed,
        store_updates_every=args.delta_t,
        run_id=args.run_id,
        expected_final_hash=ORIGINAL_FINAL_HASH,
    )
    elapsed = time.time() - start

    logger.info(
        "FedEraser prep complete in %.1f seconds (%.1f min).",
        elapsed, elapsed / 60,
    )
    logger.info(
        "Stored updates available for FedEraser configs at %s/client_updates/",
        run_dir,
    )


if __name__ == "__main__":
    main()
