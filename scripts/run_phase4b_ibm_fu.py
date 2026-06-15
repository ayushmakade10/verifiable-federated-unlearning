"""
scripts/run_phase4b_ibm_fu.py — Phase 4b: IBM FU Bundle Generation
=====================================================================

Generates 2 evidence bundles for the IBM Federated Unlearning method
(Section 9.3, Method 2). Both configs use 5 PGD epochs with different
recovery round counts (5 and 10).

Three-stage algorithm:
  1. Reconstruct client 0's local model, compute reference model
     (analytical FedAvg contribution removal).
  2. PGD unlearning: gradient ascent on client 0's data, constrained
     to an L2 ball around the reference model.
  3. Recovery: standard FedAvg rounds (via ``train()``) on the
     partition minus client 0.

Usage::

    # Generate both configs
    python scripts/run_phase4b_ibm_fu.py

    # Compatibility test: single minimal bundle + pipeline check
    python scripts/run_phase4b_ibm_fu.py --compatibility-test

    # Skip if already generated
    python scripts/run_phase4b_ibm_fu.py --skip-existing

Phase 4b of the dissertation execution roadmap (Section 9.3).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config
from data.partitioner import partition_cifar10
from evidence.bundle import build_manifest
from federation.client import train_local
from federation.trainer import train
from models.resnet import build_model
from unlearning.methods.ibm_fu import (
    compute_ball_radius,
    compute_reference_model,
    run_pgd_unlearning,
)
from utils.seeding import derive_seed

logger = logging.getLogger(__name__)


# ── IBM FU Configuration Table ──────────────────────────────────


IBM_FU_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "ibm_fu_default",
        "pgd_epochs": 5,
        "recovery_rounds": 5,
        "run_id": "phase4b/ibm_fu_default",
    },
    {
        "name": "ibm_fu_extended",
        "pgd_epochs": 5,
        "recovery_rounds": 10,
        "run_id": "phase4b/ibm_fu_extended",
    },
    {
        "name": "ibm_fu_heavy_recovery",
        "pgd_epochs": 5,
        "recovery_rounds": 50,
        "run_id": "phase4b/ibm_fu_heavy_recovery",
    },
]


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4b: Generate IBM FU unlearning evidence bundles.",
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
        help="Generate a single minimal bundle and run the verification "
             "pipeline to check for crashes.",
    )
    parser.add_argument(
        "--configs", type=str, nargs="*", default=None,
        help="Generate specific configs by name (default: all).",
    )
    return parser.parse_args()


# ── Data Helpers ─────────────────────────────────────────────────


def _build_client_dataloader(
    train_dataset: torch.utils.data.Dataset,
    client_indices: List[int],
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader over one client's data partition.

    Uses simple shuffle=True — the PGD stage is not part of the
    reproducible FL pipeline, so deterministic shuffling is not
    required.
    """
    subset = torch.utils.data.Subset(train_dataset, client_indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )


def _compute_class_histogram(
    client_indices: List[int],
    labels: np.ndarray,
    num_classes: int,
) -> List[int]:
    """Compute per-class sample counts for a client's partition."""
    client_labels = labels[client_indices]
    counter = Counter(int(lbl) for lbl in client_labels)
    return [counter.get(c, 0) for c in range(num_classes)]


# ── Stage 1: Reference Model Reconstruction ─────────────────────


def _reconstruct_client_model(
    global_state_dict: Dict[str, torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    config,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Reconstruct client 0's local model from the last FL round.

    Since per-client models are not stored, we approximate by
    running one pass of ``train_local()`` on client 0's data
    using the original global model as the starting point. This
    gives the model state that client 0 would have produced in
    the last round of federated training.

    Args:
        global_state_dict: The original global model (round 200).
        dataloader:        DataLoader over client 0's data.
        config:            Project configuration (provides optimizer
                           hyperparameters and local_epochs).
        device:            Computation device.

    Returns:
        Client 0's reconstructed local model state_dict (on CPU).
    """
    model = build_model(num_classes=config.model.num_classes)
    model.load_state_dict(global_state_dict)
    model = model.to(device)

    client_sd, n_samples = train_local(
        model=model,
        dataloader=dataloader,
        config=config,
        device=device,
    )
    logger.info(
        "Reconstructed client local model: %d samples, %d local epochs",
        n_samples, config.federation.local_epochs,
    )
    return client_sd


# ── Recovery Bundle Generation ──────────────────────────────────


def _generate_recovery_bundle(
    pgd_state_dict: Dict[str, torch.Tensor],
    partition_minus_target: Dict[int, List[int]],
    run_id: str,
    config,
    target_client: int,
    class_histogram: List[int],
    recovery_rounds: int,
) -> Path:
    """Run recovery FL rounds after PGD and produce a full evidence bundle.

    Uses the existing ``train()`` function with
    ``initial_model=pgd_output`` and ``total_rounds=recovery_rounds``.
    The resulting bundle has a valid participation log with
    ``recovery_rounds`` entries.

    After ``train()`` completes, the unlearning request is injected
    and the manifest is rebuilt to include it.
    """
    # Derive a unique seed for recovery training.
    recovery_seed = derive_seed(
        config.reproducibility.root_seed,
        f"ibm_fu_recovery_{run_id}",
    )

    run_dir = train(
        config=config,
        partition=partition_minus_target,
        run_seed=recovery_seed,
        start_round=0,
        total_rounds=recovery_rounds,
        initial_model=pgd_state_dict,
        run_id=run_id,
        save_checkpoints=False,
    )

    # Inject unlearning request.
    request = {
        "request_id": f"req_{target_client:03d}",
        "target_client_id": target_client,
        "source_run_id": "run_001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_type": "client_deletion",
        "client_class_histogram": class_histogram,
    }
    with open(run_dir / "unlearning_request.json", "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)

    # Rebuild manifest to include the unlearning request.
    build_manifest(
        run_dir=run_dir,
        run_id=run_id,
        run_seed=recovery_seed,
        total_rounds=recovery_rounds,
    )

    return run_dir


# ── IBM FU Pipeline ─────────────────────────────────────────────


def _run_ibm_fu_pipeline(
    original_sd: Dict[str, torch.Tensor],
    client_loader: torch.utils.data.DataLoader,
    config,
    pgd_epochs: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Execute Stages 1-2 of IBM FU and return the PGD output.

    This is the shared preprocessing that both configs use before
    their recovery stages diverge.

    Returns:
        PGD-unlearned state_dict (on CPU).
    """
    # Stage 1a: Reconstruct client 0's local model.
    logger.info("Stage 1a: Reconstructing client 0's local model...")
    client_sd = _reconstruct_client_model(
        global_state_dict=original_sd,
        dataloader=client_loader,
        config=config,
        device=device,
    )

    # Stage 1b: Compute reference model.
    logger.info("Stage 1b: Computing reference model (w_ref)...")
    w_ref = compute_reference_model(
        global_state_dict=original_sd,
        client_state_dict=client_sd,
        num_clients=config.data.num_clients,
    )

    # Stage 1c: Compute ball radius.
    logger.info("Stage 1c: Computing ball radius...")
    ball_radius = compute_ball_radius(
        reference_state_dict=w_ref,
        model_builder=lambda: build_model(
            num_classes=config.model.num_classes,
        ),
        num_random_inits=10,
        seed_offset=9000,
    )

    # Stage 2: PGD unlearning.
    logger.info("Stage 2: Running PGD unlearning (%d epochs)...", pgd_epochs)
    pgd_sd = run_pgd_unlearning(
        model_state_dict=w_ref,
        reference_state_dict=w_ref,
        ball_radius=ball_radius,
        dataloader=client_loader,
        model_builder=lambda: build_model(
            num_classes=config.model.num_classes,
        ),
        num_epochs=pgd_epochs,
        lr=0.01,
        momentum=0.9,
        clip_grad=5.0,
        device=device,
    )

    return pgd_sd


# ── Compatibility Test ──────────────────────────────────────────


def _run_compatibility_test(
    config,
    config_dict: Dict[str, Any],
    original_sd: Dict[str, torch.Tensor],
    client_indices: List[int],
    class_histogram: List[int],
    partition_minus_target: Dict[int, List[int]],
    train_dataset: torch.utils.data.Dataset,
    device: torch.device,
    output_base: Path,
) -> bool:
    """Generate a minimal IBM FU bundle and run verification.

    Runs the full 3-stage pipeline with 1 PGD epoch and 1 recovery
    round, then verifies at all 3 assurance levels. Returns True
    if no crashes occur.
    """
    from verification.runner import run_verification

    logger.info("=" * 60)
    logger.info("COMPATIBILITY TEST: minimal IBM FU bundle")
    logger.info("=" * 60)

    # 1. Run Stages 1-2 with 1 PGD epoch.
    client_loader = _build_client_dataloader(
        train_dataset, client_indices, config.data.batch_size,
    )
    pgd_sd = _run_ibm_fu_pipeline(
        original_sd=original_sd,
        client_loader=client_loader,
        config=config,
        pgd_epochs=1,
        device=device,
    )

    # 2. Run 1 recovery round to produce a full bundle.
    test_run_id = "phase4b/_ibm_fu_compat_test"

    recovery_seed = derive_seed(
        config.reproducibility.root_seed,
        f"ibm_fu_recovery_{test_run_id}",
    )

    test_dir_result = train(
        config=config,
        partition=partition_minus_target,
        run_seed=recovery_seed,
        start_round=0,
        total_rounds=1,
        initial_model=pgd_sd,
        run_id=test_run_id,
        save_checkpoints=False,
    )

    # Inject unlearning request.
    request = {
        "request_id": "req_000",
        "target_client_id": 0,
        "source_run_id": "run_001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_type": "client_deletion",
        "client_class_histogram": class_histogram,
    }
    with open(test_dir_result / "unlearning_request.json", "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)

    build_manifest(
        run_dir=test_dir_result,
        run_id=test_run_id,
        run_seed=recovery_seed,
        total_rounds=1,
    )

    # 3. Run verification at all 3 levels.
    gold_base = output_base / "gold" / "client_0"
    original_model_path = output_base / "run_001" / "final_model.pt"

    # Use existing calibration if available.
    calibration_path = None
    for candidate in [
        output_base / "phase4a" / "results" / "calibration.json",
        gold_base / "calibration.json",
    ]:
        if candidate.exists():
            calibration_path = candidate
            break

    all_passed = True
    levels = ["high", "strong", "basic"]

    for level in levels:
        logger.info("  Testing '%s' assurance...", level)
        try:
            verdict = run_verification(
                provider_bundle_path=test_dir_result,
                gold_base_path=gold_base,
                original_model_path=original_model_path,
                target_client=0,
                assurance_level=level,
                config=config,
                device=device,
                calibration_path=calibration_path,
                save_calibration_path=(
                    test_dir_result / "calibration.json"
                    if calibration_path is None else None
                ),
            )

            # After first run, reuse saved calibration.
            if calibration_path is None and (
                test_dir_result / "calibration.json"
            ).exists():
                calibration_path = test_dir_result / "calibration.json"

            checks_run = list(verdict.check_results.keys())
            logger.info(
                "    %s: verdict=%s, checks=%s",
                level,
                "PASS" if verdict.passed else "FAIL",
                checks_run,
            )

            # Log per-check details.
            for check_name, result in verdict.check_results.items():
                status = "PASS" if result.passed else "FAIL"
                meta_status = result.metadata.get("status", "")
                if meta_status:
                    logger.info(
                        "      %s: %s (%s)",
                        check_name, status, meta_status,
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
    """Main entry point for Phase 4b IBM FU bundle generation."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = str(REPO_ROOT / args.config)
    cfg = load_config(config_path)
    config_dict = cfg.model_dump()
    target_client = args.target_client
    output_base = Path(cfg.checkpoint.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Phase 4b: IBM FU Bundle Generation")
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

    client_indices = partition[target_client]
    logger.info(
        "Client %d has %d samples", target_client, len(client_indices),
    )

    # Partition without target client (for recovery rounds).
    partition_minus_target = {
        cid: indices for cid, indices in partition.items()
        if cid != target_client
    }

    # ── Load training dataset ────────────────────────────────────
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Compute class histogram for the target client.
    raw_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.data_root, train=True, download=False,
    )
    labels = np.array(raw_dataset.targets)
    class_histogram = _compute_class_histogram(
        client_indices, labels, cfg.model.num_classes,
    )
    logger.info("Client %d class histogram: %s", target_client, class_histogram)

    # ── Compatibility test mode ──────────────────────────────────
    if args.compatibility_test:
        success = _run_compatibility_test(
            config=cfg,
            config_dict=config_dict,
            original_sd=original_sd,
            client_indices=client_indices,
            class_histogram=class_histogram,
            partition_minus_target=partition_minus_target,
            train_dataset=train_dataset,
            device=device,
            output_base=output_base,
        )
        sys.exit(0 if success else 1)

    # ── Filter configs ───────────────────────────────────────────
    configs = IBM_FU_CONFIGS
    if args.configs is not None:
        configs = [c for c in configs if c["name"] in args.configs]

    logger.info("Generating %d IBM FU bundles:", len(configs))
    for c in configs:
        logger.info(
            "  %s: %d PGD epochs, %d recovery rounds",
            c["name"], c["pgd_epochs"], c["recovery_rounds"],
        )

    # ── Run Stages 1-2 (shared across configs) ──────────────────
    # Both configs use 5 PGD epochs, so we run the pipeline once
    # and reuse the PGD output for both recovery variants.
    logger.info("\nRunning shared Stages 1-2 (reference model + PGD)...")
    stage12_start = time.time()

    client_loader = _build_client_dataloader(
        train_dataset, client_indices, cfg.data.batch_size,
    )
    pgd_sd = _run_ibm_fu_pipeline(
        original_sd=original_sd,
        client_loader=client_loader,
        config=cfg,
        pgd_epochs=configs[0]["pgd_epochs"],
        device=device,
    )

    stage12_elapsed = time.time() - stage12_start
    logger.info("Stages 1-2 complete (%.1f seconds)", stage12_elapsed)

    # ── Stage 3: Generate recovery bundles ───────────────────────
    overall_start = time.time()

    for i, fu_cfg in enumerate(configs, 1):
        name = fu_cfg["name"]
        run_id = fu_cfg["run_id"]
        run_dir = output_base / run_id

        # Skip if already generated.
        if args.skip_existing and (run_dir / "final_model.pt").exists():
            logger.info(
                "[%d/%d] %s — already exists, skipping.",
                i, len(configs), name,
            )
            continue

        logger.info(
            "\n[%d/%d] Generating %s (recovery_rounds=%d)...",
            i, len(configs), name, fu_cfg["recovery_rounds"],
        )
        step_start = time.time()

        _generate_recovery_bundle(
            pgd_state_dict=pgd_sd,
            partition_minus_target=partition_minus_target,
            run_id=run_id,
            config=cfg,
            target_client=target_client,
            class_histogram=class_histogram,
            recovery_rounds=fu_cfg["recovery_rounds"],
        )

        elapsed = time.time() - step_start
        logger.info("  %s complete (%.1f seconds)", name, elapsed)

    overall_elapsed = time.time() - overall_start
    logger.info(
        "\nAll IBM FU bundles generated in %.1f seconds "
        "(+%.1f seconds for Stages 1-2).",
        overall_elapsed, stage12_elapsed,
    )


if __name__ == "__main__":
    main()
