"""
scripts/run_phase4a_evaluate.py — Phase 4a: Evaluation & Detection Matrix
=============================================================================

Runs the verification pipeline on all generated Phase 4a failure bundles
at all three assurance levels (basic, strong, high) and produces a
comprehensive detection matrix.

Output:
  - Per-experiment verdict JSONs in ``outputs/phase4a/results/{exp}/``
  - Aggregated detection matrix in JSON and human-readable text formats

Usage::

    # Evaluate all generated bundles
    python scripts/run_phase4a_evaluate.py

    # Evaluate specific experiments
    python scripts/run_phase4a_evaluate.py --experiments 5 6

    # Include Experiment 1 (no-unlearning) from Phase 3
    python scripts/run_phase4a_evaluate.py --include-exp1 outputs/run_001

    # Use pre-computed calibration
    python scripts/run_phase4a_evaluate.py \\
        --calibration outputs/gold/client_0/calibration.json

Phase 4a of the dissertation execution roadmap (Section 9.2).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config
from verification.runner import run_verification

logger = logging.getLogger(__name__)


# ── Assurance levels ────────────────────────────────────────────

ASSURANCE_LEVELS = ["basic", "strong", "high"]

# Checks active at each level (for matrix column headers).
LEVEL_CHECKS = {
    "basic": [
        "logit_divergence", "accuracy_parity", "evidence_consistency",
    ],
    "strong": [
        "logit_divergence", "accuracy_parity", "weight_distance",
        "checkpoint_trajectory", "evidence_consistency",
    ],
    "high": [
        "logit_divergence", "accuracy_parity", "weight_distance",
        "checkpoint_trajectory", "evidence_consistency",
    ],
}

# Short column labels for the text table.
CHECK_SHORT_NAMES = {
    "logit_divergence": "C1 KL",
    "accuracy_parity": "C2 Acc",
    "weight_distance": "C3 Cos",
    "checkpoint_trajectory": "C4 Traj",
    "evidence_consistency": "C5 Evid",
}

# Canonical experiment display order.
EXPERIMENT_ORDER = [
    "exp1_no_unlearning",
    "exp2_partial_K050",
    "exp2_partial_K100",
    "exp2_partial_K150",
    "exp3_finetune_aggressive",
    "exp3_finetune_subtle",
    "exp4_wrong_different",
    "exp4_wrong_similar",
    "exp5_rollback_R050_updated",
    "exp5_rollback_R050_stale",
    "exp5_rollback_R150_updated",
    "exp5_rollback_R150_stale",
    "exp6a_model_swap",
    "exp6b_log_edit",
    "exp6c_hash_break",
    "exp6d_checkpoint_swap",
    "exp6e_checkpoint_delete",
    "exp6f_manifest_alter",
]


# ── CLI ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4a: Evaluate failure bundles and produce "
                    "detection matrix.",
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
        "--calibration", type=str, default=None,
        help="Path to pre-computed calibration JSON. If not provided, "
             "calibration is computed from gold models on first run "
             "and saved for reuse.",
    )
    parser.add_argument(
        "--experiments", type=int, nargs="*", default=None,
        choices=[1, 2, 3, 4, 5, 6],
        help="Evaluate specific experiments by ID (default: all found).",
    )
    parser.add_argument(
        "--include-exp1", type=str, default=None,
        help="Path to the original run bundle to evaluate as Experiment 1 "
             "(no-unlearning baseline).",
    )
    return parser.parse_args()


# ── Experiment Discovery ────────────────────────────────────────


def _discover_bundles(
    output_base: Path,
    experiment_filter: Optional[List[int]],
    exp1_bundle: Optional[Path],
) -> List[Tuple[str, Path]]:
    """Find all generated Phase 4a bundles to evaluate.

    Returns a list of (experiment_name, bundle_path) tuples sorted
    in canonical experiment order.
    """
    phase4a_dir = output_base / "phase4a"
    bundles: Dict[str, Path] = {}

    # Experiment 1: no-unlearning (from Phase 3, optionally included).
    if exp1_bundle is not None:
        bundles["exp1_no_unlearning"] = exp1_bundle

    # Discover Phase 4a bundles.
    if phase4a_dir.exists():
        for entry in sorted(phase4a_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("exp"):
                # Verify it looks like a valid evidence bundle.
                has_model = (entry / "final_model.pt").exists()
                has_log = (entry / "participation_log.json").exists()
                if has_model or has_log:
                    bundles[entry.name] = entry

    # Apply experiment filter if specified.
    if experiment_filter is not None:
        prefixes = [f"exp{eid}" for eid in experiment_filter]
        bundles = {
            name: path for name, path in bundles.items()
            if any(name.startswith(p) for p in prefixes)
        }

    # Sort by canonical order, with any unknown experiments at the end.
    order_map = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}

    def sort_key(name):
        return order_map.get(name, 999)

    return [(name, bundles[name]) for name in sorted(bundles, key=sort_key)]


# ── Single-Bundle Evaluation ───────────────────────────────────


def _evaluate_bundle(
    experiment_name: str,
    bundle_path: Path,
    gold_base: Path,
    original_model_path: Path,
    target_client: int,
    config,
    device: torch.device,
    calibration_path: Optional[Path],
    save_calibration_path: Optional[Path],
    results_dir: Path,
) -> Dict[str, Any]:
    """Run verification on one bundle at all three assurance levels.

    Returns a dict mapping assurance level → verdict summary.
    """
    experiment_results: Dict[str, Any] = {}

    for level in ASSURANCE_LEVELS:
        logger.info(
            "  Verifying %s at '%s' assurance...",
            experiment_name, level,
        )

        try:
            verdict = run_verification(
                provider_bundle_path=bundle_path,
                gold_base_path=gold_base,
                original_model_path=original_model_path,
                target_client=target_client,
                assurance_level=level,
                config=config,
                device=device,
                calibration_path=calibration_path,
                save_calibration_path=save_calibration_path,
            )

            # Save individual verdict JSON.
            verdict_dir = results_dir / experiment_name
            verdict_dir.mkdir(parents=True, exist_ok=True)
            verdict_path = verdict_dir / f"verdict_{level}.json"
            verdict.save(verdict_path)

            # After first successful calibration, reuse the saved file.
            if (save_calibration_path is not None
                    and save_calibration_path.exists()):
                calibration_path = save_calibration_path
                save_calibration_path = None

            # Extract summary for the matrix.
            check_summaries = {}
            for check_name in LEVEL_CHECKS[level]:
                if check_name in verdict.check_results:
                    result = verdict.check_results[check_name]
                    check_summaries[check_name] = {
                        "passed": result.passed,
                        "deviation_ratio": result.deviation_ratio,
                        "measured_value": result.measured_value,
                        "threshold": result.threshold,
                    }

            experiment_results[level] = {
                "verdict": "PASS" if verdict.passed else "FAIL",
                "checks": check_summaries,
            }

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "  ERROR evaluating %s at %s: %s",
                experiment_name, level, exc,
            )
            experiment_results[level] = {
                "verdict": "ERROR",
                "error": str(exc),
                "checks": {},
            }

    return experiment_results


# ── Detection Matrix Formatting ─────────────────────────────────


def _format_check_cell(
    check_name: str,
    checks: Dict[str, Any],
    level: str,
) -> str:
    """Format a single check's result for the text table.

    Returns a short string like "0.81×", "✓", "✗", or "—".
    """
    if check_name not in LEVEL_CHECKS[level]:
        return "—"

    if check_name not in checks:
        return "?"

    info = checks[check_name]

    if check_name == "evidence_consistency":
        return "✓" if info["passed"] else "✗"

    ratio = info["deviation_ratio"]
    passed = info["passed"]
    marker = "" if passed else "!"
    return f"{ratio:.2f}×{marker}"


def _build_text_table(
    matrix: Dict[str, Dict[str, Any]],
) -> str:
    """Format the detection matrix as a human-readable text table."""
    all_checks = [
        "logit_divergence", "accuracy_parity", "weight_distance",
        "checkpoint_trajectory", "evidence_consistency",
    ]
    short = CHECK_SHORT_NAMES

    # Column widths.
    name_w = 35
    level_w = 7
    verdict_w = 7
    check_w = 8

    # Header.
    header = (
        f"{'Experiment':<{name_w}} "
        f"{'Level':<{level_w}} "
        f"{'Verdict':<{verdict_w}} "
    )
    header += " ".join(f"{short[c]:>{check_w}}" for c in all_checks)

    separator = "─" * len(header)

    lines = [separator, header, separator]

    # Sort experiments by canonical order.
    order_map = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}

    def sort_key(name):
        return order_map.get(name, 999)

    for exp_name in sorted(matrix, key=sort_key):
        for level in ASSURANCE_LEVELS:
            if level not in matrix[exp_name]:
                continue

            data = matrix[exp_name][level]
            verdict = data["verdict"]
            checks = data.get("checks", {})

            row = (
                f"{exp_name:<{name_w}} "
                f"{level:<{level_w}} "
                f"{verdict:<{verdict_w}} "
            )
            cells = []
            for check_name in all_checks:
                cells.append(
                    f"{_format_check_cell(check_name, checks, level):>{check_w}}"
                )
            row += " ".join(cells)
            lines.append(row)

        # Blank line between experiments.
        lines.append("")

    lines.append(separator)

    # Summary statistics.
    total = 0
    pass_count = 0
    fail_count = 0
    error_count = 0
    for exp_data in matrix.values():
        for level_data in exp_data.values():
            total += 1
            v = level_data.get("verdict", "ERROR")
            if v == "PASS":
                pass_count += 1
            elif v == "FAIL":
                fail_count += 1
            else:
                error_count += 1

    lines.append(
        f"Total: {total} verdicts — "
        f"{pass_count} PASS, {fail_count} FAIL, {error_count} ERROR"
    )
    lines.append(separator)

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for Phase 4a evaluation."""
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

    original_model_path = output_base / "run_001" / "final_model.pt"
    gold_base = output_base / "gold" / f"client_{target_client}"

    if not original_model_path.exists():
        logger.error(
            "Original model not found at %s", original_model_path,
        )
        sys.exit(1)
    if not gold_base.exists():
        logger.error("Gold models not found at %s", gold_base)
        sys.exit(1)

    # Calibration handling: load existing or compute and save.
    calibration_path = (
        Path(args.calibration) if args.calibration else None
    )
    auto_calibration_path = (
        output_base / "phase4a" / "results" / "calibration.json"
    )
    save_calibration_path = None
    if calibration_path is None:
        if auto_calibration_path.exists():
            calibration_path = auto_calibration_path
            logger.info(
                "Using auto-saved calibration from %s", calibration_path,
            )
        else:
            save_calibration_path = auto_calibration_path

    # Discover bundles to evaluate.
    exp1_bundle = Path(args.include_exp1) if args.include_exp1 else None
    bundles = _discover_bundles(output_base, args.experiments, exp1_bundle)

    if not bundles:
        logger.error(
            "No experiment bundles found. Run run_phase4a_generate.py first.",
        )
        sys.exit(1)

    logger.info("Phase 4a Evaluation")
    logger.info("  Device: %s", device)
    logger.info("  Bundles found: %d", len(bundles))
    for name, path in bundles:
        logger.info("    %s → %s", name, path)

    # Results directory.
    results_dir = output_base / "phase4a" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each bundle.
    detection_matrix: Dict[str, Dict[str, Any]] = {}
    overall_start = time.time()

    for i, (exp_name, bundle_path) in enumerate(bundles, 1):
        logger.info(
            "\n[%d/%d] Evaluating %s...", i, len(bundles), exp_name,
        )
        exp_start = time.time()

        detection_matrix[exp_name] = _evaluate_bundle(
            experiment_name=exp_name,
            bundle_path=bundle_path,
            gold_base=gold_base,
            original_model_path=original_model_path,
            target_client=target_client,
            config=cfg,
            device=device,
            calibration_path=calibration_path,
            save_calibration_path=save_calibration_path,
            results_dir=results_dir,
        )

        # After first bundle, calibration is saved — reuse it.
        if (save_calibration_path is not None
                and auto_calibration_path.exists()):
            calibration_path = auto_calibration_path
            save_calibration_path = None

        exp_elapsed = time.time() - exp_start
        logger.info("  %s done in %.1f seconds", exp_name, exp_elapsed)

    overall_elapsed = time.time() - overall_start

    # ── Save results ──────────────────────────────────────────
    matrix_json_path = results_dir / "detection_matrix.json"
    with open(matrix_json_path, "w", encoding="utf-8") as f:
        json.dump(detection_matrix, f, indent=2)
    logger.info("Detection matrix (JSON) saved to %s", matrix_json_path)

    text_table = _build_text_table(detection_matrix)
    matrix_text_path = results_dir / "detection_matrix.txt"
    with open(matrix_text_path, "w", encoding="utf-8") as f:
        f.write(text_table)
    logger.info("Detection matrix (text) saved to %s", matrix_text_path)

    # Print to console.
    print("\n" + text_table)
    print(f"\nTotal evaluation time: {overall_elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
