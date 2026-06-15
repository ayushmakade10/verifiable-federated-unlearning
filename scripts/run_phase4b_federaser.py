"""
scripts/run_phase4b_federaser.py — Phase 4b: FedEraser Bundle Generation
==========================================================================

Generates 3 evidence bundles for the FedEraser unlearning method
(Section 9.3, Method 4). All three read stored updates from the single
shared prep run (outputs/run_001_federaser_prep/), each selecting its
own round interval (Δt).

Prerequisite: run scripts/run_phase4b_federaser_prep.py first.

Configs:
    federaser_default  Δt=10  → 20 calibration rounds
    federaser_sparse   Δt=20  → 10 calibration rounds
    federaser_dense    Δt=5   → 40 calibration rounds

Usage::

    python scripts/run_phase4b_federaser.py
    python scripts/run_phase4b_federaser.py --compatibility-test
    python scripts/run_phase4b_federaser.py --configs federaser_default
    python scripts/run_phase4b_federaser.py --skip-existing
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

# pylint: disable=import-error,wrong-import-position
from config.schemas import load_config
from data.partitioner import partition_cifar10
from unlearning.methods.federaser import run_federaser
from utils.seeding import derive_seed

logger = logging.getLogger(__name__)


FEDERASER_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "federaser_default",
        "delta_t": 10,
        "run_id": "phase4b/federaser_default",
    },
    {
        "name": "federaser_sparse",
        "delta_t": 20,
        "run_id": "phase4b/federaser_sparse",
    },
    {
        "name": "federaser_dense",
        "delta_t": 5,
        "run_id": "phase4b/federaser_dense",
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4b: Generate FedEraser unlearning evidence bundles.",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to the project config YAML.",
    )
    parser.add_argument(
        "--prep-dir", type=str, default="run_001_federaser_prep",
        help="Prep run directory name (under output_dir).",
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
        help="Generate a minimal bundle (Δt=20, first 2 rounds only) "
             "and run the verification pipeline.",
    )
    parser.add_argument(
        "--configs", type=str, nargs="*", default=None,
        help="Generate specific configs by name (default: all).",
    )
    return parser.parse_args()


def _run_compatibility_test(
    cfg,
    prep_dir: Path,
    partition: Dict[int, List[int]],
    target_client: int,
    device: torch.device,
    output_base: Path,
) -> bool:
    """Generate a minimal FedEraser bundle and verify at all levels."""
    from verification.runner import run_verification

    logger.info("=" * 60)
    logger.info("COMPATIBILITY TEST: minimal FedEraser bundle (Δt=20)")
    logger.info("=" * 60)

    test_run_id = "phase4b/_federaser_compat_test"
    test_seed = derive_seed(
        cfg.reproducibility.root_seed, f"federaser_{test_run_id}",
    )

    # Δt=20 → only 10 calibration rounds, fastest config for a smoke test.
    test_dir = run_federaser(
        prep_dir=prep_dir,
        partition=partition,
        target_client=target_client,
        delta_t=20,
        config=cfg,
        run_id=test_run_id,
        run_seed=test_seed,
        device=device,
        checkpoint_every=5,
        calibration_epochs=3,
    )

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
                config=cfg,
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

            logger.info(
                "    %s: verdict=%s",
                level, "PASS" if verdict.passed else "FAIL",
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
                "    CRASH at '%s': %s", level, exc, exc_info=True,
            )
            all_passed = False

    if all_passed:
        logger.info("COMPATIBILITY TEST PASSED — no crashes at any level.")
    else:
        logger.error("COMPATIBILITY TEST FAILED — crashes detected.")
    return all_passed


def main() -> None:
    """Main entry point for Phase 4b FedEraser bundle generation."""
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
    prep_dir = output_base / args.prep_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Phase 4b: FedEraser Bundle Generation")
    logger.info("  Device: %s", device)
    logger.info("  Prep dir: %s", prep_dir)
    logger.info("  Target client: %d", target_client)

    # ── Verify prep run exists ───────────────────────────────────
    if not (prep_dir / "client_updates").exists():
        logger.error(
            "Prep run not found at %s. Run "
            "scripts/run_phase4b_federaser_prep.py first.",
            prep_dir,
        )
        sys.exit(1)

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
            cfg=cfg,
            prep_dir=prep_dir,
            partition=partition,
            target_client=target_client,
            device=device,
            output_base=output_base,
        )
        sys.exit(0 if success else 1)

    # ── Filter configs ───────────────────────────────────────────
    configs = FEDERASER_CONFIGS
    if args.configs is not None:
        configs = [c for c in configs if c["name"] in args.configs]

    logger.info("Generating %d FedEraser bundles:", len(configs))
    for c in configs:
        n_rounds = cfg.federation.num_rounds // c["delta_t"]
        logger.info(
            "  %s: Δt=%d → %d calibration rounds",
            c["name"], c["delta_t"], n_rounds,
        )

    # ── Generate bundles ─────────────────────────────────────────
    overall_start = time.time()

    for i, fe_cfg in enumerate(configs, 1):
        name = fe_cfg["name"]
        run_id = fe_cfg["run_id"]
        run_dir = output_base / run_id

        if args.skip_existing and (run_dir / "final_model.pt").exists():
            logger.info(
                "[%d/%d] %s — already exists, skipping.",
                i, len(configs), name,
            )
            continue

        logger.info(
            "\n[%d/%d] Generating %s (Δt=%d)...",
            i, len(configs), name, fe_cfg["delta_t"],
        )
        step_start = time.time()

        run_seed = derive_seed(
            cfg.reproducibility.root_seed, f"federaser_{name}",
        )

        run_federaser(
            prep_dir=prep_dir,
            partition=partition,
            target_client=target_client,
            delta_t=fe_cfg["delta_t"],
            config=cfg,
            run_id=run_id,
            run_seed=run_seed,
            device=device,
            calibration_epochs=3,
        )

        elapsed = time.time() - step_start
        logger.info("  %s complete (%.1f seconds)", name, elapsed)

    overall_elapsed = time.time() - overall_start
    logger.info(
        "\nAll FedEraser bundles generated in %.1f seconds.", overall_elapsed,
    )


if __name__ == "__main__":
    main()
