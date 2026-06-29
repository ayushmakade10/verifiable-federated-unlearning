"""
scripts/run_phase4b_quickdrop.py — QuickDrop (Method 5) Evaluation
===================================================================

Runs the verification pipeline on the three QuickDrop provider bundles
(default / extended / finetuned) under ``outputs/phase4b_quickdrop/`` and
produces a detection matrix in the same format as the cross-method Phase 4b
matrix.

Differs from ``run_phase4b_evaluate.py`` in three grounded ways:

  1. **ConvNet architecture.** Gold, original, and provider models are the
     QuickDrop ConvNet (config ``architecture: convnet`` drives ``build_model``).
     Gold/original are read from the QuickDrop sibling repo.

  2. **Computes a SEPARATE ConvNet calibration.** Rather than loading the frozen
     ResNet calibration, it computes a ConvNet calibration from the ten ConvNet
     gold models on its first pass and saves it to
     ``outputs/phase4b_quickdrop/results/calibration.json`` — the ResNet
     ``calibration.json`` is never touched.

  3. **EVALUATION_ORDER = [high, strong, basic].** High runs first so the first
     calibration computation includes ALL checks (computing basic-first would
     omit weight_distance/trajectory and KeyError at strong/high).

``--calibrate-only`` computes and saves the calibration from the first bundle's
high pass, then exits — an inspection checkpoint (gold-to-gold cosine baseline,
KL/accuracy thresholds) before the full 3x3 run.

Additive; touches no existing file.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from config.schemas import load_config  # noqa: E402
from verification.runner import run_verification  # noqa: E402

logger = logging.getLogger(__name__)

ASSURANCE_LEVELS = ["basic", "strong", "high"]      # display order
EVALUATION_ORDER = ["high", "strong", "basic"]      # compute order (full calib first)

LEVEL_CHECKS = {
    "basic": ["logit_divergence", "accuracy_parity", "evidence_consistency"],
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

QUICKDROP_CONFIG_ORDER = [
    "quickdrop_default",
    "quickdrop_extended",
    "quickdrop_finetuned",
]


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4b: Evaluate QuickDrop (ConvNet) bundles."
    )
    parser.add_argument(
        "--config", type=str, default="config/quickdrop.yaml",
        help="QuickDrop config (architecture=convnet).",
    )
    parser.add_argument(
        "--quickdrop-root", type=str,
        default=str(REPO_ROOT.parent / "quickdrop-main" / "quickdrop-main"),
        help="QuickDrop sibling repo (holds vfu_outputs gold/original).",
    )
    parser.add_argument(
        "--target-client", type=int, default=0,
    )
    parser.add_argument(
        "--configs", type=str, nargs="*", default=None,
        help="Evaluate specific configs by name (default: all found).",
    )
    parser.add_argument(
        "--calibrate-only", action="store_true",
        help="Compute+save the ConvNet calibration from the first bundle's "
             "high pass, then exit (inspection checkpoint).",
    )
    return parser.parse_args()


# ── Bundle Discovery ─────────────────────────────────────────────


def _discover_bundles(
    output_base: Path,
    config_filter: Optional[List[str]],
) -> List[Tuple[str, Path]]:
    """Find QuickDrop bundles under outputs/phase4b_quickdrop/."""
    root = output_base / "phase4b_quickdrop"
    bundles: Dict[str, Path] = {}
    if not root.exists():
        return []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_") or entry.name == "results":
            continue
        if not (entry / "final_model.pt").exists():
            continue
        bundles[entry.name] = entry
    if config_filter is not None:
        bundles = {n: p for n, p in bundles.items() if n in config_filter}
    order_map = {name: i for i, name in enumerate(QUICKDROP_CONFIG_ORDER)}
    return [(n, bundles[n]) for n in sorted(bundles, key=lambda n: order_map.get(n, 999))]


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
    levels: List[str],
) -> Tuple[Dict[str, Any], Optional[Path], Optional[Path]]:
    """Verify one bundle across the given assurance levels.

    Returns (config_results, calibration_path, save_calibration_path) so the
    caller can thread the calibration paths to subsequent bundles after the
    first computes+saves it.
    """
    config_results: Dict[str, Any] = {}

    for level in levels:
        logger.info("  Verifying %s at '%s' assurance...", config_name, level)
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

            verdict_dir = results_dir / config_name
            verdict_dir.mkdir(parents=True, exist_ok=True)
            verdict.save(verdict_dir / f"verdict_{level}.json")

            # Reuse calibration after first save.
            if (save_calibration_path is not None
                    and save_calibration_path.exists()):
                calibration_path = save_calibration_path
                save_calibration_path = None

            check_summaries = {}
            for check_name in LEVEL_CHECKS[level]:
                if check_name in verdict.check_results:
                    r = verdict.check_results[check_name]
                    check_summaries[check_name] = {
                        "passed": r.passed,
                        "deviation_ratio": r.deviation_ratio,
                        "measured_value": r.measured_value,
                        "threshold": r.threshold,
                        "metadata_status": r.metadata.get("status", ""),
                    }
            config_results[level] = {
                "verdict": "PASS" if verdict.passed else "FAIL",
                "checks": check_summaries,
            }

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("  ERROR evaluating %s at %s: %s",
                         config_name, level, exc, exc_info=True)
            config_results[level] = {
                "verdict": "ERROR", "error": str(exc), "checks": {},
            }

    return config_results, calibration_path, save_calibration_path


# ── Detection Matrix Formatting ──────────────────────────────────


def _format_check_cell(check_name: str, checks: Dict[str, Any], level: str) -> str:
    if check_name not in LEVEL_CHECKS[level]:
        return "—"
    if check_name not in checks:
        return "?"
    info = checks[check_name]
    if info.get("metadata_status", "") == "insufficient_evidence":
        return "insuf"
    if check_name == "evidence_consistency":
        return "✓" if info["passed"] else "✗"
    marker = "" if info["passed"] else "!"
    return f"{info['deviation_ratio']:.2f}×{marker}"


def _build_text_table(matrix: Dict[str, Dict[str, Any]]) -> str:
    all_checks = [
        "logit_divergence", "accuracy_parity", "weight_distance",
        "checkpoint_trajectory", "evidence_consistency",
    ]
    short = CHECK_SHORT_NAMES
    name_w, level_w, verdict_w, check_w = 30, 7, 7, 8

    header = (f"{'Config':<{name_w}} {'Level':<{level_w}} {'Verdict':<{verdict_w}} ")
    header += " ".join(f"{short[c]:>{check_w}}" for c in all_checks)
    sep = "─" * len(header)
    lines = [sep, "Phase 4b — QuickDrop Detection Matrix", sep, header, sep]

    order_map = {name: i for i, name in enumerate(QUICKDROP_CONFIG_ORDER)}

    def sort_key(name):
        return order_map.get(name, 999)

    for cfg_name in sorted(matrix, key=sort_key):
        for level in ASSURANCE_LEVELS:
            if level not in matrix[cfg_name]:
                continue
            data = matrix[cfg_name][level]
            checks = data.get("checks", {})
            row = (f"{cfg_name:<{name_w}} {level:<{level_w}} "
                   f"{data['verdict']:<{verdict_w}} ")
            row += " ".join(
                f"{_format_check_cell(c, checks, level):>{check_w}}"
                for c in all_checks
            )
            lines.append(row)
        lines.append("")
    lines.append(sep)

    total = pass_count = fail_count = error_count = 0
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
    lines.append(f"Total: {total} verdicts — "
                 f"{pass_count} PASS, {fail_count} FAIL, {error_count} ERROR")
    lines.append(sep)

    lines.append("")
    lines.append("Per-Check Deviation Ratios (high assurance):")
    lines.append(sep)
    ratio_header = f"{'Config':<{name_w}} "
    ratio_header += " ".join(f"{short[c]:>{check_w}}" for c in all_checks)
    lines.append(ratio_header)
    lines.append(sep)
    for cfg_name in sorted(matrix, key=sort_key):
        if "high" not in matrix[cfg_name]:
            continue
        checks = matrix[cfg_name]["high"].get("checks", {})
        row = f"{cfg_name:<{name_w}} "
        row += " ".join(
            f"{_format_check_cell(c, checks, 'high'):>{check_w}}"
            for c in all_checks
        )
        lines.append(row)
    lines.append(sep)
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(str(REPO_ROOT / args.config))
    if cfg.model.architecture != "convnet":
        logger.error("Config architecture must be 'convnet', got '%s'",
                     cfg.model.architecture)
        sys.exit(1)

    target_client = args.target_client
    output_base = Path(cfg.checkpoint.output_dir)
    quickdrop_root = Path(args.quickdrop_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ConvNet gold + original live in the QuickDrop sibling repo.
    gold_base = quickdrop_root / "vfu_outputs" / "gold_convnet"
    original_model_path = (
        quickdrop_root / "vfu_outputs" / "provider" / "original_model.pt"
    )
    if not original_model_path.exists():
        logger.error("ConvNet original not found at %s", original_model_path)
        sys.exit(1)
    if not gold_base.exists():
        logger.error("ConvNet gold not found at %s", gold_base)
        sys.exit(1)

    # SEPARATE ConvNet calibration — never the ResNet calibration.json.
    results_dir = output_base / "phase4b_quickdrop" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    convnet_calibration = results_dir / "calibration.json"

    calibration_path: Optional[Path] = None
    save_calibration_path: Optional[Path] = None
    if convnet_calibration.exists():
        calibration_path = convnet_calibration
        logger.info("Using existing ConvNet calibration: %s", convnet_calibration)
    else:
        save_calibration_path = convnet_calibration
        logger.info("No ConvNet calibration yet; will compute and save to %s",
                    convnet_calibration)

    bundles = _discover_bundles(output_base, args.configs)
    if not bundles:
        logger.error("No QuickDrop bundles found under %s. "
                     "Run build_quickdrop_bundles.py first.",
                     output_base / "phase4b_quickdrop")
        sys.exit(1)

    logger.info("Phase 4b QuickDrop Evaluation")
    logger.info("  Device: %s", device)
    logger.info("  Gold (ConvNet): %s", gold_base)
    logger.info("  Bundles found: %d", len(bundles))
    for name, path in bundles:
        logger.info("    %s → %s", name, path)

    # ── Calibrate-only: compute on the first bundle's high pass, then exit ──
    if args.calibrate_only:
        if calibration_path is not None:
            logger.info("Calibration already exists at %s — nothing to compute. "
                        "(Delete it to force recompute.)", calibration_path)
            return
        first_name, first_path = bundles[0]
        logger.info("[calibrate-only] computing ConvNet calibration from "
                    "first bundle (%s) high pass...", first_name)
        _evaluate_bundle(
            config_name=first_name, bundle_path=first_path, gold_base=gold_base,
            original_model_path=original_model_path, target_client=target_client,
            config=cfg, device=device, calibration_path=None,
            save_calibration_path=save_calibration_path, results_dir=results_dir,
            levels=["high"],
        )
        if convnet_calibration.exists():
            logger.info("[calibrate-only] saved: %s", convnet_calibration)
        else:
            logger.error("[calibrate-only] calibration was NOT written.")
        return

    # ── Full evaluation ──────────────────────────────────────────
    detection_matrix: Dict[str, Dict[str, Any]] = {}
    overall_start = time.time()
    for i, (cfg_name, bundle_path) in enumerate(bundles, 1):
        logger.info("\n[%d/%d] Evaluating %s...", i, len(bundles), cfg_name)
        step_start = time.time()
        results, calibration_path, save_calibration_path = _evaluate_bundle(
            config_name=cfg_name, bundle_path=bundle_path, gold_base=gold_base,
            original_model_path=original_model_path, target_client=target_client,
            config=cfg, device=device, calibration_path=calibration_path,
            save_calibration_path=save_calibration_path, results_dir=results_dir,
            levels=EVALUATION_ORDER,
        )
        detection_matrix[cfg_name] = results
        logger.info("  %s done in %.1f seconds", cfg_name, time.time() - step_start)

    logger.info("\nTotal evaluation time: %.1f seconds",
                time.time() - overall_start)

    matrix_json_path = results_dir / "detection_matrix.json"
    with open(matrix_json_path, "w", encoding="utf-8") as f:
        json.dump(detection_matrix, f, indent=2)
    logger.info("Detection matrix (JSON) saved to %s", matrix_json_path)

    text_table = _build_text_table(detection_matrix)
    with open(results_dir / "detection_matrix.txt", "w", encoding="utf-8") as f:
        f.write(text_table)
    print("\n" + text_table)


if __name__ == "__main__":
    main()
