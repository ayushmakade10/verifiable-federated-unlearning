"""
scripts/run_verification.py — CLI Entry Point for Verification
================================================================

Runs the full verification pipeline: loads a provider's evidence bundle,
calibrates thresholds from gold-standard models (or loads pre-computed
calibration), executes all active checks at the chosen assurance level,
and outputs the verdict + fidelity report.

Usage:
    # Basic assurance (Checks 1, 2, 5):
    python scripts/run_verification.py \\
        --provider-bundle outputs/run_unlearned \\
        --assurance basic

    # Strong assurance (Checks 1–5):
    python scripts/run_verification.py \\
        --provider-bundle outputs/run_unlearned \\
        --assurance strong

    # With pre-computed calibration:
    python scripts/run_verification.py \\
        --provider-bundle outputs/run_unlearned \\
        --assurance strong \\
        --calibration outputs/gold/client_0/calibration.json

Phase 3 of the dissertation execution roadmap (Section 9.2).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config
from verification.runner import run_verification


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the verification script."""
    parser = argparse.ArgumentParser(
        description="Run the full verification pipeline on a provider's evidence bundle.",
    )
    parser.add_argument(
        "--provider-bundle", type=str, required=True,
        help="Path to the provider's evidence bundle directory.",
    )
    parser.add_argument(
        "--original-run", type=str, default="run_001",
        help="Run ID for the original training (default: run_001).",
    )
    parser.add_argument(
        "--target-client", type=int, default=0,
        help="Target client ID for unlearning (default: 0).",
    )
    parser.add_argument(
        "--assurance", type=str, default="strong",
        choices=["basic", "strong", "high"],
        help="Assurance level (default: strong).",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to the project config YAML.",
    )
    parser.add_argument(
        "--calibration", type=str, default=None,
        help="Path to pre-computed calibration JSON. If not provided, "
             "calibration is computed from gold models.",
    )
    parser.add_argument(
        "--save-calibration", type=str, default=None,
        help="Save computed calibration to this path for reuse.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the verdict/fidelity report JSON. "
             "Defaults to outputs/verification/client_{k}/{bundle}/verdict.json.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the verification pipeline from the command line."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = str(REPO_ROOT / args.config)
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_base = Path(cfg.checkpoint.output_dir)
    provider_bundle = Path(args.provider_bundle)
    original_model_path = output_base / args.original_run / "final_model.pt"
    gold_base = output_base / "gold" / f"client_{args.target_client}"

    calibration_path = Path(args.calibration) if args.calibration else None
    save_calibration_path = (
        Path(args.save_calibration) if args.save_calibration else None
    )
    output_path = (
        Path(args.output)
        if args.output
        else output_base / "verification" / f"client_{args.target_client}"
             / provider_bundle.name / "verdict.json"
    )

    # ── Run verification ─────────────────────────────────────────
    verdict = run_verification(
        provider_bundle_path=provider_bundle,
        gold_base_path=gold_base,
        original_model_path=original_model_path,
        target_client=args.target_client,
        assurance_level=args.assurance,
        config=cfg,
        device=device,
        calibration_path=calibration_path,
        save_calibration_path=save_calibration_path,
    )

    # ── Output ───────────────────────────────────────────────────
    verdict.save(output_path)
    verdict.print_summary()
    logging.getLogger(__name__).info("Verdict saved to %s", output_path)


if __name__ == "__main__":
    main()
