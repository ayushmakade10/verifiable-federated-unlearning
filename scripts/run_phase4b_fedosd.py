"""
scripts/run_phase4b_fedosd.py — Phase 4b: FedOSD Bundle Generation
=====================================================================

Generates 3 evidence bundles for the FedOSD unlearning method
(Section 9.3, Method 3). Each config varies the number of
unlearning and recovery rounds.

Unlike SGA/IBM FU, each FedOSD config runs independently — different
unlearning round counts produce different gradient histories and
model trajectories. No sharing is possible between configs.

Usage::

    # Generate all 3 configs
    python scripts/run_phase4b_fedosd.py

    # Compatibility test: minimal single-round bundle + pipeline check
    python scripts/run_phase4b_fedosd.py --compatibility-test

    # Specific configs only
    python scripts/run_phase4b_fedosd.py --configs fedosd_default fedosd_extended_recovery

    # Skip already-generated bundles
    python scripts/run_phase4b_fedosd.py --skip-existing

Phase 4b of the dissertation execution roadmap (Section 9.3).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config
from data.partitioner import partition_cifar10
from unlearning.methods.fedosd import run_fedosd
from utils.seeding import derive_seed

logger = logging.getLogger(__name__)


# ── FedOSD Configuration Table ──────────────────────────────────


FEDOSD_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "fedosd_default",
        "unlearning_rounds": 10,
        "recovery_rounds": 10,
        "run_id": "phase4b/fedosd_default",
    },
    {
        "name": "fedosd_extended_recovery",
        "unlearning_rounds": 10,
        "recovery_rounds": 20,
        "run_id": "phase4b/fedosd_extended_recovery",
    },
    {
        "name": "fedosd_extended_unlearning",
        "unlearning_rounds": 20,
        "recovery_rounds": 10,
        "run_id": "phase4b/fedosd_extended_unlearning",
    },
]


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4b: Generate FedOSD unlearning evidence bundles.",
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
        help="Skip bundles that already have a final_model.pt.",
    )
    parser.add_argument(
        "--compatibility-test", action="store_true",
        help="Generate a minimal single-round bundle and run the "
             "verification pipeline to check for crashes.",
    )
    parser.add_argument(
        "--configs", type=str, nargs="*", default=None,
        help="Generate specific configs by name (default: all).",
    )
    return parser.parse_args()


# ── Compatibility Test ──────────────────────────────────────────


def _run_compatibility_test(
    config,
    original_sd: Dict[str, torch.Tensor],
    partition: Dict[int, List[int]],
    target_client: int,
    device: torch.device,
    output_base: Path,
) -> bool:
    """Generate a minimal FedOSD bundle and verify at all levels.

    Uses 1 unlearning round + 1 recovery round to produce the
    smallest possible valid bundle, then runs the verification
    pipeline at all three assurance levels.

    Returns True if no crashes occur.
    """
    from verification.runner import run_verification

    logger.info("=" * 60)
    logger.info("COMPATIBILITY TEST: minimal FedOSD bundle")
    logger.info("=" * 60)

    test_run_id = "phase4b/_fedosd_compat_test"
    test_seed = derive_seed(
        config.reproducibility.root_seed,
        f"fedosd_{test_run_id}",
    )

    test_dir = run_fedosd(
        global_state_dict=original_sd,
        partition=partition,
        target_client=target_client,
        config=config,
        num_unlearning_rounds=1,
        num_recovery_rounds=1,
        run_id=test_run_id,
        run_seed=test_seed,
        device=device,
        checkpoint_every=1,
    )

    # Run verification at all 3 levels.
    gold_base = output_base / "gold" / "client_0"
    original_model_path = output_base / "run_001" / "final_model.pt"

    calibration_path = None
    for candidate in [
        output_base / "phase4a" / "results" / "calibration.json",
        gold_base / "calibration.json",
    ]:
        if candidate.exists():
            calibration_path = candidate
            break

    all_passed = True

    for level in ["basic", "strong", "high"]:
        logger.info("  Testing '%s' assurance...", level)
        try:
            verdict = run_verification(
                provider_bundle_path=test_dir,
                gold_base_path=gold_base,
                original_model_path=original_model_path,
                target_client=target_client,
                assurance_level=level,
                config=config,
                device=device,
                calibration_path=calibration_path,
                save_calibration_path=(
                    test_dir / "calibration.json"
                    if calibration_path is None else None
                ),
            )

            if calibration_path is None and (
                test_dir / "calibration.json"
            ).exists():
                calibration_path = test_dir / "calibration.json"

            checks_run = list(verdict.check_results.keys())
            logger.info(
                "    %s: verdict=%s, checks=%s",
                level,
                "PASS" if verdict.passed else "FAIL",
                checks_run,
            )

            for check_name, result in verdict.check_results.items():
                status = "PASS" if result.passed else "FAIL"
                meta_status = result.metadata.get("status", "")
                if meta_status:
                    logger.info(
                        "      %s: %s (%s)", check_name, status, meta_status,
                    )
                else:
                    logger.info(
                        "      %s: %s (ratio=%.3f)",
                        check_name, status, result.deviation_ratio,
                    )

        except Exception as exc:
            logger.error(
                "    CRASH at '%s' assurance: %s", level, exc,
                exc_info=True,
            )
            all_passed = False

    if all_passed:
        logger.info("COMPATIBILITY TEST PASSED — no crashes at any level.")
    else:
        logger.error("COMPATIBILITY TEST FAILED — crashes detected.")

    return all_passed


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for Phase 4b FedOSD bundle generation."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = str(REPO_ROOT / args.config)
    cfg = load_config(config_path)
    target_client = args.target_client
    output_base = Path(cfg.checkpoint.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Phase 4b: FedOSD Bundle Generation")
    logger.info("  Device: %s", device)
    logger.info("  Target client: %d", target_client)

    # ── Load original model ──────────────────────────────────────
    original_model_path = output_base / "run_001" / "final_model.pt"
    if not original_model_path.exists():
        logger.error("Original model not found at %s", original_model_path)
        sys.exit(1)

    original_sd = torch.load(original_model_path, weights_only=True)
    logger.info("Original model loaded from %s", original_model_path)

    # ── Reconstruct partition ────────────────────────────────────
    partition_seed = derive_seed(cfg.reproducibility.root_seed, "partition")
    partition = partition_cifar10(
        num_clients=cfg.data.num_clients,
        alpha=cfg.data.alpha,
        seed=partition_seed,
    )
    logger.info(
        "Client %d has %d samples",
        target_client, len(partition[target_client]),
    )

    # ── Compatibility test mode ──────────────────────────────────
    if args.compatibility_test:
        success = _run_compatibility_test(
            config=cfg,
            original_sd=original_sd,
            partition=partition,
            target_client=target_client,
            device=device,
            output_base=output_base,
        )
        sys.exit(0 if success else 1)

    # ── Filter configs ───────────────────────────────────────────
    configs = FEDOSD_CONFIGS
    if args.configs is not None:
        configs = [c for c in configs if c["name"] in args.configs]

    logger.info("Generating %d FedOSD bundles:", len(configs))
    for c in configs:
        logger.info(
            "  %s: %d unlearning + %d recovery rounds",
            c["name"], c["unlearning_rounds"], c["recovery_rounds"],
        )

    # ── Generate bundles ─────────────────────────────────────────
    overall_start = time.time()

    for i, osd_cfg in enumerate(configs, 1):
        name = osd_cfg["name"]
        run_id = osd_cfg["run_id"]
        run_dir = output_base / run_id

        # Skip if already generated.
        if args.skip_existing and (run_dir / "final_model.pt").exists():
            logger.info(
                "[%d/%d] %s — already exists, skipping.",
                i, len(configs), name,
            )
            continue

        logger.info(
            "\n[%d/%d] Generating %s...", i, len(configs), name,
        )
        step_start = time.time()

        run_seed = derive_seed(
            cfg.reproducibility.root_seed,
            f"fedosd_{name}",
        )

        run_fedosd(
            global_state_dict=original_sd,
            partition=partition,
            target_client=target_client,
            config=cfg,
            num_unlearning_rounds=osd_cfg["unlearning_rounds"],
            num_recovery_rounds=osd_cfg["recovery_rounds"],
            run_id=run_id,
            run_seed=run_seed,
            device=device,
        )

        elapsed = time.time() - step_start
        logger.info("  %s complete (%.1f seconds)", name, elapsed)

    overall_elapsed = time.time() - overall_start
    logger.info(
        "\nAll FedOSD bundles generated in %.1f seconds.", overall_elapsed,
    )


if __name__ == "__main__":
    main()
