"""
scripts/run_phase4b_sga.py — Phase 4b: SGA Bundle Generation
================================================================

Generates 6 evidence bundles for the Simple SGA unlearning method
(Section 9.3, Method 1). Three SGA-only variants (no recovery) and
three with 3 FL recovery rounds afterward.

SGA-only bundles are assembled manually (no FL training). Recovery
bundles use the existing ``train()`` function with
``initial_model=sga_output, total_rounds=3``.

Usage::

    # Generate all 6 bundles
    python scripts/run_phase4b_sga.py

    # Generate only non-recovery (fast, minutes)
    python scripts/run_phase4b_sga.py --skip-recovery

    # Generate only recovery variants
    python scripts/run_phase4b_sga.py --only-recovery

    # Compatibility test: single minimal bundle + pipeline check
    python scripts/run_phase4b_sga.py --compatibility-test

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
from evidence.participation_log import ParticipationLog
from federation.trainer import train
from unlearning.methods.sga import run_sga
from utils.seeding import derive_seed

logger = logging.getLogger(__name__)


# ── SGA Configuration Table ─────────────────────────────────────


SGA_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "sga_light",
        "epochs": 1,
        "lr": 0.01,
        "recovery_rounds": 0,
        "run_id": "phase4b/sga_light",
    },
    {
        "name": "sga_light_recovered",
        "epochs": 1,
        "lr": 0.01,
        "recovery_rounds": 3,
        "run_id": "phase4b/sga_light_recovered",
    },
    {
        "name": "sga_moderate",
        "epochs": 5,
        "lr": 0.01,
        "recovery_rounds": 0,
        "run_id": "phase4b/sga_moderate",
    },
    {
        "name": "sga_moderate_recovered",
        "epochs": 5,
        "lr": 0.01,
        "recovery_rounds": 3,
        "run_id": "phase4b/sga_moderate_recovered",
    },
    {
        "name": "sga_aggressive",
        "epochs": 10,
        "lr": 0.01,
        "recovery_rounds": 0,
        "run_id": "phase4b/sga_aggressive",
    },
    {
        "name": "sga_aggressive_recovered",
        "epochs": 10,
        "lr": 0.01,
        "recovery_rounds": 3,
        "run_id": "phase4b/sga_aggressive_recovered",
    },
]


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4b: Generate SGA unlearning evidence bundles.",
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
        "--skip-recovery", action="store_true",
        help="Generate only non-recovery SGA bundles.",
    )
    parser.add_argument(
        "--only-recovery", action="store_true",
        help="Generate only recovery SGA bundles.",
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
    return parser.parse_args()


# ── Data Helpers ─────────────────────────────────────────────────


def _build_client_dataloader(
    train_dataset: torch.utils.data.Dataset,
    client_indices: List[int],
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader over one client's data partition.

    Uses no shuffling randomness — SGA is not part of the
    reproducible FL pipeline, so deterministic shuffling is
    not required. Simple shuffle=True suffices.
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


# ── Bundle Assembly (non-recovery) ───────────────────────────────


def _assemble_sga_bundle(
    sga_state_dict: Dict[str, torch.Tensor],
    run_dir: Path,
    run_id: str,
    config_dict: Dict[str, Any],
    target_client: int,
    class_histogram: List[int],
    source_run_id: str = "run_001",
) -> Path:
    """Assemble a minimal evidence bundle for a non-recovery SGA run.

    Creates:
      - final_model.pt
      - config.yaml (frozen copy)
      - participation_log.json (0 rounds, valid metadata)
      - unlearning_request.json
      - manifest.json (hashes of all the above)

    No checkpoints directory is created (SGA has no FL rounds).
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save final model.
    torch.save(sga_state_dict, run_dir / "final_model.pt")

    # 2. Frozen config.
    import yaml
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # 3. Participation log — 0 FL rounds.
    log = ParticipationLog(
        run_id=run_id,
        run_seed=0,  # No FL rounds, seed is irrelevant.
        num_clients=config_dict["data"]["num_clients"],
        participation_rate=config_dict["federation"]["participation_rate"],
        rounds=[],
    )
    log.save(run_dir / "participation_log.json")

    # 4. Unlearning request.
    request = {
        "request_id": f"req_{target_client:03d}",
        "target_client_id": target_client,
        "source_run_id": source_run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_type": "client_deletion",
        "client_class_histogram": class_histogram,
    }
    with open(run_dir / "unlearning_request.json", "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)

    # 5. Manifest (hashes all files above).
    build_manifest(
        run_dir=run_dir,
        run_id=run_id,
        run_seed=0,
        total_rounds=0,
    )

    return run_dir


# ── Recovery Bundle (via trainer) ────────────────────────────────


def _generate_recovery_bundle(
    sga_state_dict: Dict[str, torch.Tensor],
    partition_minus_target: Dict[int, List[int]],
    run_id: str,
    config,
    target_client: int,
    class_histogram: List[int],
    recovery_rounds: int = 3,
) -> Path:
    """Run recovery FL rounds after SGA and produce a full evidence bundle.

    Uses the existing ``train()`` function with ``initial_model=sga_output``
    and ``total_rounds=recovery_rounds``. The resulting bundle has a valid
    participation log with ``recovery_rounds`` entries.

    After ``train()`` completes, the unlearning request is injected into
    the bundle and the manifest is rebuilt to include it.
    """
    # Derive a unique seed for recovery training.
    recovery_seed = derive_seed(
        config.reproducibility.root_seed,
        f"sga_recovery_{run_id}",
    )

    run_dir = train(
        config=config,
        partition=partition_minus_target,
        run_seed=recovery_seed,
        start_round=0,
        total_rounds=recovery_rounds,
        initial_model=sga_state_dict,
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


# ── Compatibility Test ───────────────────────────────────────────


def _run_compatibility_test(
    config,
    config_dict: Dict[str, Any],
    original_sd: Dict[str, torch.Tensor],
    client_indices: List[int],
    class_histogram: List[int],
    train_dataset: torch.utils.data.Dataset,
    device: torch.device,
    output_base: Path,
) -> bool:
    """Generate a minimal SGA bundle and run verification to check for crashes.

    Returns True if all 5 checks complete at all 3 assurance levels.
    """
    from verification.runner import run_verification

    logger.info("=" * 60)
    logger.info("COMPATIBILITY TEST: minimal SGA bundle")
    logger.info("=" * 60)

    # 1. Run minimal SGA (1 epoch).
    loader = _build_client_dataloader(train_dataset, client_indices, 64)
    sga_sd = run_sga(
        model_state_dict=original_sd,
        dataloader=loader,
        num_epochs=1,
        learning_rate=0.01,
        device=device,
    )

    # 2. Assemble minimal bundle.
    test_dir = output_base / "phase4b" / "_compatibility_test"
    _assemble_sga_bundle(
        sga_state_dict=sga_sd,
        run_dir=test_dir,
        run_id="phase4b/_compatibility_test",
        config_dict=config_dict,
        target_client=0,
        class_histogram=class_histogram,
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
                provider_bundle_path=test_dir,
                gold_base_path=gold_base,
                original_model_path=original_model_path,
                target_client=0,
                assurance_level=level,
                config=config,
                device=device,
                calibration_path=calibration_path,
                save_calibration_path=(
                    test_dir / "calibration.json"
                    if calibration_path is None else None
                ),
            )

            # After first run, reuse saved calibration.
            if calibration_path is None and (test_dir / "calibration.json").exists():
                calibration_path = test_dir / "calibration.json"

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
    """Main entry point for Phase 4b SGA bundle generation."""
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

    logger.info("Phase 4b: SGA Bundle Generation")
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
            train_dataset=train_dataset,
            device=device,
            output_base=output_base,
        )
        sys.exit(0 if success else 1)

    # ── Filter configs ───────────────────────────────────────────
    configs = SGA_CONFIGS
    if args.skip_recovery:
        configs = [c for c in configs if c["recovery_rounds"] == 0]
    elif args.only_recovery:
        configs = [c for c in configs if c["recovery_rounds"] > 0]

    logger.info("Generating %d SGA bundles:", len(configs))
    for c in configs:
        logger.info(
            "  %s: %d epochs, lr=%.3f, recovery=%d",
            c["name"], c["epochs"], c["lr"], c["recovery_rounds"],
        )

    # ── Generate each bundle ─────────────────────────────────────
    overall_start = time.time()

    for i, sga_cfg in enumerate(configs, 1):
        name = sga_cfg["name"]
        run_id = sga_cfg["run_id"]
        run_dir = output_base / run_id

        # Skip if already generated.
        if args.skip_existing and (run_dir / "final_model.pt").exists():
            logger.info("[%d/%d] %s — already exists, skipping.", i, len(configs), name)
            continue

        logger.info(
            "\n[%d/%d] Generating %s (epochs=%d, lr=%.3f, recovery=%d)...",
            i, len(configs), name,
            sga_cfg["epochs"], sga_cfg["lr"], sga_cfg["recovery_rounds"],
        )
        step_start = time.time()

        # Run SGA.
        client_loader = _build_client_dataloader(
            train_dataset, client_indices, cfg.data.batch_size,
        )
        sga_sd = run_sga(
            model_state_dict=original_sd,
            dataloader=client_loader,
            num_epochs=sga_cfg["epochs"],
            learning_rate=sga_cfg["lr"],
            device=device,
            num_classes=cfg.model.num_classes,
        )

        if sga_cfg["recovery_rounds"] == 0:
            # Non-recovery: assemble bundle manually.
            _assemble_sga_bundle(
                sga_state_dict=sga_sd,
                run_dir=run_dir,
                run_id=run_id,
                config_dict=config_dict,
                target_client=target_client,
                class_histogram=class_histogram,
            )
        else:
            # Recovery: use train() for FL rounds.
            _generate_recovery_bundle(
                sga_state_dict=sga_sd,
                partition_minus_target=partition_minus_target,
                run_id=run_id,
                config=cfg,
                target_client=target_client,
                class_histogram=class_histogram,
                recovery_rounds=sga_cfg["recovery_rounds"],
            )

        elapsed = time.time() - step_start
        logger.info("  %s complete (%.1f seconds)", name, elapsed)

    overall_elapsed = time.time() - overall_start
    logger.info(
        "\nAll SGA bundles generated in %.1f seconds.", overall_elapsed,
    )


if __name__ == "__main__":
    main()
