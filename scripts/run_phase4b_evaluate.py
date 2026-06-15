"""
scripts/run_phase4b_evaluate.py — Phase 4b: SGA Evaluation & Detection Matrix
================================================================================

Runs the verification pipeline on all generated Phase 4b SGA bundles
at all three assurance levels (basic, strong, high) and produces a
detection matrix comparable to Phase 4a's output.

Uses the **frozen calibration from Phase 3/4a** — never recomputes
thresholds. The calibration is the fixed reference against which all
methods are evaluated.

Output:
  - Per-config verdict JSONs in ``outputs/phase4b/results/{config}/``
  - Aggregated detection matrix in JSON and human-readable text formats
  - Summary table with per-check deviation ratios

Usage::

    # Evaluate all generated SGA bundles
    python scripts/run_phase4b_evaluate.py

    # Specify calibration file explicitly
    python scripts/run_phase4b_evaluate.py \\
        --calibration outputs/phase4a/results/calibration.json

Phase 4b of the dissertation execution roadmap (Section 9.3).
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


# ── Constants ────────────────────────────────────────────────────

ASSURANCE_LEVELS = ["basic", "strong", "high"]

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

CHECK_SHORT_NAMES = {
    "logit_divergence": "C1 KL",
    "accuracy_parity": "C2 Acc",
    "weight_distance": "C3 Cos",
    "checkpoint_trajectory": "C4 Traj",
    "evidence_consistency": "C5 Evid",
}

# Canonical display order for SGA configs.
SGA_CONFIG_ORDER = [
    "sga_light",
    "sga_light_recovered",
    "sga_moderate",
    "sga_moderate_recovered",
    "sga_aggressive",
    "sga_aggressive_recovered",
]


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4b: Evaluate SGA bundles and produce "
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
        help="Path to pre-computed calibration JSON. Searches standard "
             "locations if not provided.",
    )
    parser.add_argument(
        "--configs", type=str, nargs="*", default=None,
        help="Evaluate specific configs by name (default: all found).",
    )
    return parser.parse_args()


# ── Bundle Discovery ─────────────────────────────────────────────


def _discover_bundles(
    output_base: Path,
    config_filter: Optional[List[str]],
) -> List[Tuple[str, Path]]:
    """Find all generated Phase 4b SGA bundles to evaluate.

    Scans ``outputs/phase4b/`` for directories containing a
    ``final_model.pt``. Excludes the compatibility test directory.

    Returns (config_name, bundle_path) tuples in canonical order.
    """
    phase4b_dir = output_base / "phase4b"
    bundles: Dict[str, Path] = {}

    if not phase4b_dir.exists():
        return []

    for entry in sorted(phase4b_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Skip internal directories.
        if entry.name.startswith("_") or entry.name == "results":
            continue
        if not (entry / "final_model.pt").exists():
            continue
        bundles[entry.name] = entry

    # Apply filter if specified.
    if config_filter is not None:
        bundles = {
            name: path for name, path in bundles.items()
            if name in config_filter
        }

    # Sort by canonical order.
    order_map = {name: i for i, name in enumerate(SGA_CONFIG_ORDER)}

    def sort_key(name):
        return order_map.get(name, 999)

    return [(name, bundles[name]) for name in sorted(bundles, key=sort_key)]


# ── Single-Bundle Evaluation ────────────────────────────────────


def _evaluate_bundle(
    config_name: str,
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
    """Run verification on one SGA bundle at all three assurance levels.

    Returns a dict mapping assurance level to verdict summary.
    """
    config_results: Dict[str, Any] = {}

    for level in ASSURANCE_LEVELS:
        logger.info(
            "  Verifying %s at '%s' assurance...",
            config_name, level,
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
            verdict_dir = results_dir / config_name
            verdict_dir.mkdir(parents=True, exist_ok=True)
            verdict_path = verdict_dir / f"verdict_{level}.json"
            verdict.save(verdict_path)

            # Reuse calibration after first save.
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
                        "metadata_status": result.metadata.get("status", ""),
                    }

            config_results[level] = {
                "verdict": "PASS" if verdict.passed else "FAIL",
                "checks": check_summaries,
            }

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "  ERROR evaluating %s at %s: %s",
                config_name, level, exc,
                exc_info=True,
            )
            config_results[level] = {
                "verdict": "ERROR",
                "error": str(exc),
                "checks": {},
            }

    return config_results


# ── Detection Matrix Formatting ──────────────────────────────────


def _format_check_cell(
    check_name: str,
    checks: Dict[str, Any],
    level: str,
) -> str:
    """Format a single check's result for the text table."""
    if check_name not in LEVEL_CHECKS[level]:
        return "—"

    if check_name not in checks:
        return "?"

    info = checks[check_name]

    # Checks with a non-standard status (insufficient evidence, etc.).
    meta_status = info.get("metadata_status", "")
    if meta_status == "insufficient_evidence":
        return "insuf"

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

    name_w = 30
    level_w = 7
    verdict_w = 7
    check_w = 8

    header = (
        f"{'Config':<{name_w}} "
        f"{'Level':<{level_w}} "
        f"{'Verdict':<{verdict_w}} "
    )
    header += " ".join(f"{short[c]:>{check_w}}" for c in all_checks)

    separator = "─" * len(header)

    lines = [
        separator,
        "Phase 4b — SGA Detection Matrix",
        separator,
        header,
        separator,
    ]

    order_map = {name: i for i, name in enumerate(SGA_CONFIG_ORDER)}

    def sort_key(name):
        return order_map.get(name, 999)

    for cfg_name in sorted(matrix, key=sort_key):
        for level in ASSURANCE_LEVELS:
            if level not in matrix[cfg_name]:
                continue

            data = matrix[cfg_name][level]
            verdict = data["verdict"]
            checks = data.get("checks", {})

            row = (
                f"{cfg_name:<{name_w}} "
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

        lines.append("")

    lines.append(separator)

    # Summary statistics.
    total = 0
    pass_count = 0
    fail_count = 0
    error_count = 0
    for cfg_data in matrix.values():
        for level_data in cfg_data.values():
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

    # Deviation ratio summary table.
    lines.append("")
    lines.append("Per-Check Deviation Ratios (high assurance):")
    lines.append(separator)

    ratio_header = f"{'Config':<{name_w}} "
    ratio_header += " ".join(f"{short[c]:>{check_w}}" for c in all_checks)
    lines.append(ratio_header)
    lines.append(separator)

    for cfg_name in sorted(matrix, key=sort_key):
        if "high" not in matrix[cfg_name]:
            continue
        checks = matrix[cfg_name]["high"].get("checks", {})
        row = f"{cfg_name:<{name_w}} "
        cells = []
        for check_name in all_checks:
            cells.append(
                f"{_format_check_cell(check_name, checks, 'high'):>{check_w}}"
            )
        row += " ".join(cells)
        lines.append(row)

    lines.append(separator)

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for Phase 4b SGA evaluation."""
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
        logger.error("Original model not found at %s", original_model_path)
        sys.exit(1)
    if not gold_base.exists():
        logger.error("Gold models not found at %s", gold_base)
        sys.exit(1)

    # ── Find calibration ─────────────────────────────────────────
    calibration_path = (
        Path(args.calibration) if args.calibration else None
    )
    auto_calibration_path = (
        output_base / "phase4b" / "results" / "calibration.json"
    )
    save_calibration_path = None

    if calibration_path is None:
        # Search standard locations.
        for candidate in [
            output_base / "phase4a" / "results" / "calibration.json",
            gold_base / "calibration.json",
        ]:
            if candidate.exists():
                calibration_path = candidate
                logger.info("Using calibration from %s", calibration_path)
                break

    if calibration_path is None:
        if auto_calibration_path.exists():
            calibration_path = auto_calibration_path
        else:
            save_calibration_path = auto_calibration_path
            logger.info(
                "No pre-computed calibration found. Will compute and save.",
            )

    # ── Discover bundles ─────────────────────────────────────────
    bundles = _discover_bundles(output_base, args.configs)

    if not bundles:
        logger.error(
            "No SGA bundles found. Run run_phase4b_sga.py first.",
        )
        sys.exit(1)

    logger.info("Phase 4b SGA Evaluation")
    logger.info("  Device: %s", device)
    logger.info("  Bundles found: %d", len(bundles))
    for name, path in bundles:
        logger.info("    %s → %s", name, path)

    # ── Results directory ────────────────────────────────────────
    results_dir = output_base / "phase4b" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate each bundle ─────────────────────────────────────
    detection_matrix: Dict[str, Dict[str, Any]] = {}
    overall_start = time.time()

    for i, (cfg_name, bundle_path) in enumerate(bundles, 1):
        logger.info(
            "\n[%d/%d] Evaluating %s...", i, len(bundles), cfg_name,
        )
        step_start = time.time()

        detection_matrix[cfg_name] = _evaluate_bundle(
            config_name=cfg_name,
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

        step_elapsed = time.time() - step_start
        logger.info("  %s done in %.1f seconds", cfg_name, step_elapsed)

    overall_elapsed = time.time() - overall_start

    # ── Save results ─────────────────────────────────────────────
    matrix_json_path = results_dir / "detection_matrix.json"
    with open(matrix_json_path, "w", encoding="utf-8") as f:
        json.dump(detection_matrix, f, indent=2)
    logger.info("Detection matrix (JSON) saved to %s", matrix_json_path)

    text_table = _build_text_table(detection_matrix)
    matrix_text_path = results_dir / "detection_matrix.txt"
    with open(matrix_text_path, "w", encoding="utf-8") as f:
        f.write(text_table)
    logger.info("Detection matrix (text) saved to %s", matrix_text_path)

    print("\n" + text_table)
    print(f"\nTotal evaluation time: {overall_elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
