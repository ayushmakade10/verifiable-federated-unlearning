"""
verification/runner.py — Verification Pipeline Orchestrator
==============================================================

Top-level entry point that ties together all verification components:

  1. Load provider evidence bundle (model, checkpoints, log, manifest)
  2. Load gold models + original model
  3. Build probe set from unlearning request histogram
  4. Calibrate thresholds (or load pre-computed calibration)
  5. Select active checks based on assurance level
  6. Run each active check
  7. Compose verdict by conjunction
  8. Assemble and return the fidelity report

This module is called by the CLI script (scripts/run_verification.py)
and can also be used programmatically.

Specification references: Sections 4.1–4.8, 9.2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision  # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error

from config.schemas import ProjectConfig
from models.resnet import build_model
from verification.calibration import (
    CalibrationBundle,
    calibrate_accuracy_parity,
    calibrate_logit_divergence,
    calibrate_trajectory,
    calibrate_weight_distance,
)
from verification.checks import ASSURANCE_CHECKS, CheckResult
from verification.checks.check_accuracy_parity import (
    run_check as run_accuracy_parity,
)
from verification.checks.check_evidence import (
    run_check as run_evidence_consistency,
)
from verification.checks.check_logit_divergence import (
    run_check as run_logit_divergence,
)
from verification.checks.check_trajectory import (
    run_check as run_trajectory,
)
from verification.checks.check_weight_distance import (
    run_check as run_weight_distance,
)
from verification.probe_set import build_deterministic_probe_set
from verification.verdict import VerificationVerdict, compose_verdict

logger = logging.getLogger(__name__)


def run_verification(
    provider_bundle_path: Path,
    gold_base_path: Path,
    original_model_path: Path,
    target_client: int,
    assurance_level: str,
    config: ProjectConfig,
    device: torch.device,
    calibration_path: Optional[Path] = None,
    save_calibration_path: Optional[Path] = None,
) -> VerificationVerdict:
    """Execute the full verification pipeline.

    Args:
        provider_bundle_path: Path to the provider's evidence bundle
            directory (e.g. outputs/run_unlearned/).
        gold_base_path: Base path for gold models
            (e.g. outputs/gold/client_0/).
        original_model_path: Path to the original model
            (e.g. outputs/run_001/final_model.pt).
        target_client: Target client ID for unlearning.
        assurance_level: "basic", "strong", or "high".
        config: Project configuration.
        device: Device for inference.
        calibration_path: If provided, load pre-computed calibration
            from this path instead of recalculating.
        save_calibration_path: If provided, save computed calibration
            to this path for reuse.

    Returns:
        VerificationVerdict with binary decision and fidelity report.
    """
    level = assurance_level.lower()
    active_checks = ASSURANCE_CHECKS[level]
    num_trials = config.gold_standard.num_trials

    logger.info("Starting verification at '%s' assurance", level)
    logger.info("Active checks: %s", active_checks)

    # ── Load provider model ──────────────────────────────────────
    logger.info("Loading provider model...")
    provider_sd = torch.load(
        provider_bundle_path / "final_model.pt", weights_only=True,
    )
    provider_model = build_model(num_classes=config.model.num_classes)
    provider_model.load_state_dict(provider_sd)
    provider_model = provider_model.to(device)

    # ── Load original model ──────────────────────────────────────
    logger.info("Loading original model...")
    original_sd = torch.load(original_model_path, weights_only=True)

    # ── Load gold models ─────────────────────────────────────────
    logger.info("Loading %d gold models...", num_trials)
    gold_state_dicts, gold_models = _load_gold_models(
        gold_base_path, num_trials, config.model.num_classes, device,
    )

    # ── Load gold checkpoints (for Check 4) ──────────────────────
    gold_checkpoint_sets = []
    if "checkpoint_trajectory" in active_checks:
        gold_checkpoint_sets = _load_gold_checkpoints(
            gold_base_path, num_trials,
        )
        logger.info(
            "Loaded %d gold checkpoint sequences for trajectory analysis",
            len(gold_checkpoint_sets),
        )

    # ── Load provider checkpoints (for Check 4) ──────────────────
    provider_checkpoints = []
    if "checkpoint_trajectory" in active_checks:
        provider_checkpoints = _load_provider_checkpoints(
            provider_bundle_path,
        )
        logger.info(
            "Loaded %d provider checkpoints", len(provider_checkpoints),
        )

    # ── Build probe set ──────────────────────────────────────────
    probe_loader, full_loader = _build_evaluation_loaders(
        provider_bundle_path, config,
    )

    # ── Calibrate or load calibration ────────────────────────────
    calibration = _get_calibration(
        calibration_path, save_calibration_path,
        gold_models=gold_models,
        gold_state_dicts=gold_state_dicts,
        original_sd=original_sd,
        gold_checkpoint_sets=gold_checkpoint_sets,
        probe_loader=probe_loader,
        full_loader=full_loader,
        device=device,
        target_client=target_client,
        active_checks=active_checks,
    )

    # ── Run active checks ────────────────────────────────────────
    check_results: Dict[str, CheckResult] = {}

    if "logit_divergence" in active_checks:
        logger.info("Running Check 1: Logit Divergence...")
        cal = calibration.checks["logit_divergence"]
        cal_full = calibration.checks.get(
            "logit_divergence_full",
            calibration.checks["logit_divergence"],
        )
        check_results["logit_divergence"] = run_logit_divergence(
            provider_model, gold_models,
            probe_loader, full_loader, device,
            cal.pairwise_matrix, cal_full.pairwise_matrix,
        )

    if "accuracy_parity" in active_checks:
        logger.info("Running Check 2: Accuracy Parity...")
        cal = calibration.checks["accuracy_parity"]
        cal_full = calibration.checks.get(
            "accuracy_parity_full",
            calibration.checks["accuracy_parity"],
        )
        check_results["accuracy_parity"] = run_accuracy_parity(
            provider_model, gold_models,
            probe_loader, full_loader, device,
            cal.pairwise_matrix, cal_full.pairwise_matrix,
        )

    if "weight_distance" in active_checks:
        logger.info("Running Check 3: Weight Distance...")
        cal = calibration.checks["weight_distance"]
        check_results["weight_distance"] = run_weight_distance(
            provider_sd, gold_state_dicts, original_sd,
            cal.pairwise_matrix,
        )

    if "checkpoint_trajectory" in active_checks:
        logger.info("Running Check 4: Checkpoint Trajectory...")
        cal = calibration.checks.get("checkpoint_trajectory")
        if cal is not None and gold_checkpoint_sets:
            check_results["checkpoint_trajectory"] = run_trajectory(
                provider_checkpoints, gold_checkpoint_sets,
                cal.pairwise_matrix,
            )
        else:
            check_results["checkpoint_trajectory"] = CheckResult(
                check_name="checkpoint_trajectory",
                passed=False,
                measured_value=0.0,
                threshold=0.0,
                deviation_ratio=0.0,
                metadata={
                    "status": "insufficient_evidence",
                    "reason": "No gold checkpoint sequences available for calibration",
                },
            )

    if "evidence_consistency" in active_checks:
        logger.info("Running Check 5: Evidence Consistency...")
        check_results["evidence_consistency"] = run_evidence_consistency(
            provider_bundle_path, level,
            config.federation.num_rounds,
            config.data.num_clients,
            config.federation.participation_rate,
            config.checkpoint.save_every_n_rounds,
        )

    # ── Compose verdict ──────────────────────────────────────────
    logger.info("Composing verdict...")
    verdict = compose_verdict(check_results, level)
    return verdict


# ── Internal Helpers ─────────────────────────────────────────────


def _load_gold_models(
    gold_base_path: Path,
    num_trials: int,
    num_classes: int,
    device: torch.device,
) -> tuple[List[Dict[str, torch.Tensor]], List[torch.nn.Module]]:
    """Load all gold model state_dicts and instantiated models."""
    state_dicts = []
    models = []
    for trial in range(num_trials):
        path = gold_base_path / f"trial_{trial:02d}" / "final_model.pt"
        sd = torch.load(path, weights_only=True)
        state_dicts.append(sd)
        model = build_model(num_classes=num_classes)
        model.load_state_dict(sd)
        model = model.to(device)
        models.append(model)
    return state_dicts, models


def _load_gold_checkpoints(
    gold_base_path: Path,
    num_trials: int,
) -> List[List[Dict[str, torch.Tensor]]]:
    """Load checkpoint sequences from gold trials that have them.

    Only trials with a checkpoints/ directory containing ≥2 files
    are included (typically trials 0–2).
    """
    checkpoint_sets = []
    for trial in range(num_trials):
        ckpt_dir = gold_base_path / f"trial_{trial:02d}" / "checkpoints"
        if not ckpt_dir.exists():
            continue
        ckpt_files = sorted(ckpt_dir.glob("*.pt"))
        if len(ckpt_files) < 2:
            continue
        ckpts = [
            torch.load(f, weights_only=True) for f in ckpt_files
        ]
        checkpoint_sets.append(ckpts)
        logger.info(
            "  Gold trial %d: %d checkpoints loaded", trial, len(ckpts),
        )
    return checkpoint_sets


def _load_provider_checkpoints(
    bundle_path: Path,
) -> List[Dict[str, torch.Tensor]]:
    """Load provider checkpoint state_dicts in round order."""
    ckpt_dir = bundle_path / "checkpoints"
    if not ckpt_dir.exists():
        return []
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    return [torch.load(f, weights_only=True) for f in ckpt_files]


def _build_evaluation_loaders(
    provider_bundle_path: Path,
    config: ProjectConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build probe set and full test set DataLoaders.

    Reads the client_class_histogram from unlearning_request.json
    in the provider's bundle. Falls back to uniform weighting if
    the file is absent.
    """
    # Load test dataset.
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data.data_root,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )

    # Read class histogram from unlearning request.
    request_path = provider_bundle_path / "unlearning_request.json"
    if request_path.exists():
        with open(request_path, encoding="utf-8") as f:
            request = json.load(f)
        histogram = request.get(
            "client_class_histogram",
            [1] * config.model.num_classes,
        )
    else:
        logger.warning(
            "No unlearning_request.json found in %s; using uniform probe set",
            provider_bundle_path,
        )
        histogram = [1] * config.model.num_classes

    return build_deterministic_probe_set(
        test_dataset, histogram,
        batch_size=config.data.batch_size,
    )


def _get_calibration(
    load_path: Optional[Path],
    save_path: Optional[Path],
    *,
    gold_models: List[torch.nn.Module],
    gold_state_dicts: List[Dict[str, torch.Tensor]],
    original_sd: Dict[str, torch.Tensor],
    gold_checkpoint_sets: List[List[Dict[str, torch.Tensor]]],
    probe_loader: torch.utils.data.DataLoader,
    full_loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_client: int,
    active_checks: List[str],
) -> CalibrationBundle:
    """Load or compute the calibration bundle."""
    if load_path is not None and load_path.exists():
        logger.info("Loading pre-computed calibration from %s", load_path)
        return CalibrationBundle.load(load_path)

    logger.info("Computing calibration from gold models...")
    bundle = CalibrationBundle(
        num_gold_models=len(gold_models),
        target_client=target_client,
    )

    if "logit_divergence" in active_checks:
        logger.info("  Calibrating Check 1 (probe set)...")
        bundle.checks["logit_divergence"] = calibrate_logit_divergence(
            gold_models, probe_loader, device, "logit_divergence",
        )
        logger.info("  Calibrating Check 1 (full test)...")
        bundle.checks["logit_divergence_full"] = calibrate_logit_divergence(
            gold_models, full_loader, device, "logit_divergence_full",
        )

    if "accuracy_parity" in active_checks:
        logger.info("  Calibrating Check 2 (probe set)...")
        bundle.checks["accuracy_parity"] = calibrate_accuracy_parity(
            gold_models, probe_loader, device, "accuracy_parity",
        )
        logger.info("  Calibrating Check 2 (full test)...")
        bundle.checks["accuracy_parity_full"] = calibrate_accuracy_parity(
            gold_models, full_loader, device, "accuracy_parity_full",
        )

    if "weight_distance" in active_checks:
        logger.info("  Calibrating Check 3 (directional cosine)...")
        bundle.checks["weight_distance"] = calibrate_weight_distance(
            gold_state_dicts, original_sd,
        )

    if "checkpoint_trajectory" in active_checks and gold_checkpoint_sets:
        logger.info("  Calibrating Check 4 (trajectory)...")
        bundle.checks["checkpoint_trajectory"] = calibrate_trajectory(
            gold_checkpoint_sets,
        )

    if save_path is not None:
        bundle.save(save_path)

    return bundle
