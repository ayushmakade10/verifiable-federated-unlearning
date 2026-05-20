"""
federation/trainer.py — Resumable FedAvg Training Loop
========================================================

Single function, parameterised entry point. No separate code paths
for resumed runs. Supports all six failure behaviours from Section 5.1
via the same set of parameters.

Calling patterns (from Section 7.1):
  - Original run:
      train(cfg, full_partition, root_seed)
  - Gold trial 3:
      train(cfg, partition_without_client_k,
            derive_seed(root_seed, "gold_retrain_3"))
  - Partial retraining:
      train(cfg, partition_without_client_k, root_seed,
            start_round=150, initial_model=ckpt_150,
            initial_log=log_up_to_150)
  - Fine-tuning masquerade:
      train(cfg, partition_without_client_k, some_seed,
            total_rounds=20, initial_model=original_final_model)

Output: Evidence bundle at outputs/{run_id}/ per Section 6.1.

Usage:
    from federation.trainer import train
    from config.schemas import load_config
    from data.partitioner import partition_cifar10
    from utils.seeding import derive_seed

    cfg = load_config("config/default.yaml")
    partition = partition_cifar10(50, 0.3,
                    derive_seed(cfg.reproducibility.root_seed, "partition"))
    train(cfg, partition, cfg.reproducibility.root_seed)
"""

from __future__ import annotations

import copy
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from config.schemas import ProjectConfig
from evidence.bundle import build_manifest, save_frozen_config_from_dict
from evidence.hashing import hash_model
from evidence.participation_log import ParticipationLog
from federation.aggregation import fed_avg
from federation.client import train_local
from models.resnet import build_model
from utils.seeding import derive_seed, seed_worker, set_global_seed

logger = logging.getLogger(__name__)


# ── Data Loading Helpers ─────────────────────────────────────────


def _get_cifar10_transforms():
    """Standard CIFAR-10 transforms for ResNet-18.

    Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize.
    Test:  Normalize only.

    Normalization values are the per-channel mean and std of the
    CIFAR-10 training set — standard practice.
    """
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
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform


def _make_client_dataloader(
    dataset: torch.utils.data.Dataset,
    indices: List[int],
    batch_size: int,
    generator_seed: int,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for a single client's data subset.

    Uses a dedicated torch.Generator seeded per-round-per-client for
    reproducible shuffling, and the seed_worker function for
    per-worker reproducibility.

    Args:
        dataset:        The full training dataset (with transforms applied).
        indices:        This client's sample indices from the partition.
        batch_size:     Batch size from config.
        generator_seed: Seed for the DataLoader's generator (derived from
                        run_seed, round number, and client ID).

    Returns:
        A DataLoader over the client's data subset.
    """
    subset = torch.utils.data.Subset(dataset, indices)
    g = torch.Generator()
    g.manual_seed(generator_seed)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # In-process for determinism on all platforms.
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )


def _make_test_dataloader(
    data_root: str,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Create the CIFAR-10 test set DataLoader.

    Created once and reused across all rounds — not recreated per round.

    Args:
        data_root:  Root directory for CIFAR-10 data.
        batch_size: Batch size for evaluation.

    Returns:
        A DataLoader over the CIFAR-10 test set.
    """
    _, test_transform = _get_cifar10_transforms()
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


# ── Evaluation ───────────────────────────────────────────────────


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate a model on a test DataLoader.

    Args:
        model:      The model to evaluate (set to eval mode internally).
        dataloader: The test DataLoader.
        device:     Device for inference.

    Returns:
        (accuracy, loss) as floats.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return accuracy, avg_loss


# ── Main Training Loop ───────────────────────────────────────────


def train(
    config: ProjectConfig,
    partition: Dict[int, List[int]],
    run_seed: int,
    start_round: int = 0,
    total_rounds: Optional[int] = None,
    initial_model: Optional[Dict[str, torch.Tensor]] = None,
    initial_log: Optional[List[Dict[str, Any]]] = None,
    run_id: str = "run_001",
    save_checkpoints: bool = True,
    checkpoint_every: Optional[int] = None,
) -> Path:
    """Resumable FedAvg training loop.

    This is the single entry point for ALL training scenarios: original
    runs, gold-standard retraining, partial retraining, and fine-tuning
    masquerade. The behaviour is controlled entirely by parameters —
    no separate code paths for resumed runs.

    Args:
        config:           Project configuration (hyperparameters, paths).
        partition:        Client-to-indices mapping. Already modified if
                          needed (e.g. target client removed for gold runs).
        run_seed:         Seed for all randomness in THIS run. For gold
                          trial k, pass derive_seed(root, "gold_retrain_k").
        start_round:      Which round to begin at (0 for fresh runs,
                          >0 for resumed/partial retraining).
        total_rounds:     Which round to end at. Defaults to
                          config.federation.num_rounds.
        initial_model:    None → fresh init via build_model() with seeded
                          weights. Or a state_dict to resume from.
        initial_log:      None → empty log. Or a list of prior round entries
                          for resumed runs (prepended to the new log).
        run_id:           Output directory name under config.checkpoint.output_dir.
        save_checkpoints: Whether to save intermediate checkpoint files.
                          Set False for gold trials 3–9 (final model only).
        checkpoint_every: Checkpoint interval in rounds. Defaults to
                          config.checkpoint.save_every_n_rounds. Ignored
                          if save_checkpoints is False.

    Returns:
        Path to the run's output directory (the evidence bundle).

    Raises:
        ValueError: If start_round >= total_rounds or partition is empty.
    """
    # ── Setup ────────────────────────────────────────────────────
    if total_rounds is None:
        total_rounds = config.federation.num_rounds
    if checkpoint_every is None:
        checkpoint_every = config.checkpoint.save_every_n_rounds

    if start_round >= total_rounds:
        raise ValueError(
            f"start_round ({start_round}) must be < total_rounds ({total_rounds})"
        )
    if not partition:
        raise ValueError("Partition is empty — no clients to train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(config.checkpoint.output_dir) / run_id
    ckpt_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    if save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Starting training: run_id=%s, rounds %d→%d, seed=%d, device=%s",
        run_id, start_round, total_rounds, run_seed, device,
    )

    # ── Save frozen config ───────────────────────────────────────
    config_dict = config.model_dump()
    save_frozen_config_from_dict(config_dict, run_dir)

    # ── Model initialisation ─────────────────────────────────────
    if initial_model is None:
        # Fresh model with deterministic init.
        init_seed = derive_seed(run_seed, "model_init")
        set_global_seed(init_seed)
        model = build_model(num_classes=config.model.num_classes)
        logger.info("Fresh model initialised with seed %d", init_seed)
    else:
        model = build_model(num_classes=config.model.num_classes)
        model.load_state_dict(initial_model)
        logger.info("Model loaded from initial_model state_dict")

    model = model.to(device)

    # ── Dataset & test DataLoader (created once) ─────────────────
    # Seed the PRNG before dataset creation so the state is identical
    # regardless of whether we took the fresh-init or resumed path.
    # With real CIFAR-10 (loaded from disk) this is a no-op, but it
    # ensures correctness if the dataset constructor touches the PRNG.
    data_seed = derive_seed(run_seed, "data_loading")
    set_global_seed(data_seed)

    train_transform, _ = _get_cifar10_transforms()
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data.data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_loader = _make_test_dataloader(
        data_root=config.data.data_root,
        batch_size=config.data.batch_size * 2,  # Larger batches for eval speed.
    )

    # ── Participation log ────────────────────────────────────────
    available_clients = sorted(partition.keys())
    num_clients = len(available_clients)
    num_selected = max(1, round(num_clients * config.federation.participation_rate))

    log = ParticipationLog(
        run_id=run_id,
        run_seed=run_seed,
        num_clients=num_clients,
        participation_rate=config.federation.participation_rate,
        rounds=initial_log,
        available_clients=available_clients,
    )

    # ── Training loop ────────────────────────────────────────────
    for r in range(start_round, total_rounds):
        round_start = time.time()

        # 1. Hash global model BEFORE aggregation.
        global_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        hash_pre = hash_model(global_sd)

        # 2. Client selection — seed derived from round number, not loop index.
        selection_seed = derive_seed(run_seed, f"client_selection_round_{r}")
        rng = random.Random(selection_seed)
        selected = sorted(rng.sample(available_clients, num_selected))

        # 3. Local training for each selected client.
        client_updates: List[tuple[Dict[str, torch.Tensor], int]] = []
        samples_per_client: Dict[int, int] = {}

        for client_id in selected:
            # Per-client, per-round seed for DataLoader reproducibility.
            client_dl_seed = derive_seed(
                run_seed, f"dataloader_round_{r}_client_{client_id}"
            )
            loader = _make_client_dataloader(
                dataset=train_dataset,
                indices=partition[client_id],
                batch_size=config.data.batch_size,
                generator_seed=client_dl_seed,
            )

            # Copy global model for this client's local training.
            # Uses deepcopy instead of build_model() + load_state_dict()
            # to avoid consuming global PRNG via Kaiming init. The
            # deepcopy produces an exact replica on the same device.
            client_model = copy.deepcopy(model)

            # Set per-client seed for local training reproducibility
            # (dropout, etc. — ResNet-18 doesn't use dropout but this
            # ensures correctness for any future architecture).
            local_train_seed = derive_seed(
                run_seed, f"local_train_round_{r}_client_{client_id}"
            )
            set_global_seed(local_train_seed)

            updated_sd, n_samples = train_local(
                model=client_model,
                dataloader=loader,
                config=config,
                device=device,
            )
            client_updates.append((updated_sd, n_samples))
            samples_per_client[client_id] = n_samples

        # 4. Aggregate via FedAvg.
        aggregated_sd = fed_avg(client_updates)
        model.load_state_dict(aggregated_sd)

        # 5. Hash global model AFTER aggregation.
        hash_post = hash_model(aggregated_sd)

        # 6. Evaluate on test set.
        test_acc, test_loss = _evaluate(model, test_loader, device)

        # 7. Log this round.
        log.add_round(
            round_id=r,
            selection_seed=selection_seed,
            selected_clients=selected,
            num_samples_per_client=samples_per_client,
            global_model_hash_pre=hash_pre,
            global_model_hash_post=hash_post,
            test_accuracy=test_acc,
            test_loss=test_loss,
        )

        # 8. Save checkpoint if at interval.
        if save_checkpoints and (r + 1) % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"round_{r + 1:03d}.pt"
            torch.save(aggregated_sd, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

        elapsed = time.time() - round_start
        logger.info(
            "Round %d/%d — acc: %.4f, loss: %.4f (%.1fs)",
            r + 1, total_rounds, test_acc, test_loss, elapsed,
        )

    # ── Finalisation ─────────────────────────────────────────────

    # Save final model.
    final_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    final_path = run_dir / "final_model.pt"
    torch.save(final_sd, final_path)
    logger.info("Final model saved: %s", final_path)

    # Save participation log.
    log_path = run_dir / "participation_log.json"
    log.save(log_path)
    logger.info("Participation log saved: %s", log_path)

    # Build manifest (hashes all files in the bundle).
    end_time = datetime.now(timezone.utc).isoformat()
    build_manifest(
        run_dir=run_dir,
        run_id=run_id,
        run_seed=run_seed,
        total_rounds=total_rounds,
        dataset=config.data.dataset,
        architecture=config.model.architecture,
        start_time=start_time,
        end_time=end_time,
    )
    logger.info("Manifest saved: %s", run_dir / "manifest.json")

    logger.info("Training complete. Evidence bundle at: %s", run_dir)
    return run_dir
