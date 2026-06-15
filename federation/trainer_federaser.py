"""
federation/trainer_federaser.py — FedEraser Prep Training (with storage)
==========================================================================

A focused wrapper around the FedAvg training loop that additionally
stores per-client model updates every Δt rounds, for use by the
FedEraser unlearning method (Phase 4b, Method 4).

CRITICAL — hash preservation:
    This function reproduces ``federation.trainer.train()`` EXACTLY,
    with one addition: a read-only storage hook that saves each
    participating client's update (client_state − global_state) to
    disk before aggregation. The hook performs only pure reads
    (state_dict diffs → disk) and consumes NO PRNG and mutates NO
    model state, so the final model is bit-identical to ``train()``.
    A hash assertion at the end verifies this.

The existing ``train()`` function is NOT modified. This wrapper exists
precisely so that the default training path is untouched.

Storage layout:
    {run_dir}/client_updates/round_{r:03d}/client_{id:02d}.pt

where each .pt is a state_dict of (client_update tensors) for that
client at that round.

Specification reference: Section 9.3, Method 4 (FedEraser prep).
"""

from __future__ import annotations

import copy
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision

from config.schemas import ProjectConfig
from evidence.bundle import build_manifest, save_frozen_config_from_dict
from evidence.hashing import hash_model
from evidence.participation_log import ParticipationLog
from federation.aggregation import fed_avg
from federation.client import train_local

# Reuse train()'s private helpers rather than duplicating them. This
# guarantees identical dataset/transform/eval behaviour and therefore
# identical training dynamics.
from federation.trainer import (
    _evaluate,
    _get_cifar10_transforms,
    _make_client_dataloader,
    _make_test_dataloader,
)
from models.resnet import build_model
from utils.seeding import derive_seed, set_global_seed

logger = logging.getLogger(__name__)


def train_with_update_storage(
    config: ProjectConfig,
    partition: Dict[int, List[int]],
    run_seed: int,
    store_updates_every: int,
    run_id: str = "run_001_federaser_prep",
    total_rounds: Optional[int] = None,
    expected_final_hash: Optional[str] = None,
) -> Path:
    """Run FedAvg training identical to train(), storing per-client updates.

    The ONLY difference from ``federation.trainer.train()`` is the
    storage hook that saves per-client updates every
    ``store_updates_every`` rounds. The hook is a pure read and does
    not affect training dynamics; the final model hash is bit-identical
    to a plain ``train()`` run with the same seed and config.

    Args:
        config:              Project configuration.
        partition:           Full client partition (all 50 clients).
        run_seed:            Seed for all randomness (use the SAME seed
                             as the original run_001 training).
        store_updates_every: Δt — store updates at rounds Δt, 2Δt, ...
        run_id:              Output directory name (isolated from run_001).
        total_rounds:        Defaults to config.federation.num_rounds.
        expected_final_hash: If provided, assert the final model hash
                             matches this value (hash-preservation check).

    Returns:
        Path to the prep run's output directory.

    Raises:
        ValueError: If start_round >= total_rounds or partition is empty.
        AssertionError: If expected_final_hash is set and doesn't match.
    """
    if total_rounds is None:
        total_rounds = config.federation.num_rounds
    if not partition:
        raise ValueError("Partition is empty — no clients to train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(config.checkpoint.output_dir) / run_id
    ckpt_dir = run_dir / "checkpoints"
    updates_dir = run_dir / "client_updates"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    updates_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_every = config.checkpoint.save_every_n_rounds
    start_time = datetime.now(timezone.utc).isoformat()

    logger.info(
        "FedEraser prep: run_id=%s, %d rounds, Δt=%d, seed=%d, device=%s",
        run_id, total_rounds, store_updates_every, run_seed, device,
    )

    # ── Save frozen config ───────────────────────────────────────
    config_dict = config.model_dump()
    save_frozen_config_from_dict(config_dict, run_dir)

    # ── Model initialisation (identical to train() lines 280-291) ──
    init_seed = derive_seed(run_seed, "model_init")
    set_global_seed(init_seed)
    model = build_model(num_classes=config.model.num_classes)
    model = model.to(device)
    logger.info("Fresh model initialised with seed %d", init_seed)

    # ── Dataset & test loader (identical to train() lines 298-311) ──
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
        batch_size=config.data.batch_size * 2,
    )

    # ── Participation log ────────────────────────────────────────
    available_clients = sorted(partition.keys())
    num_clients = len(available_clients)
    num_selected = max(
        1, round(num_clients * config.federation.participation_rate),
    )

    log = ParticipationLog(
        run_id=run_id,
        run_seed=run_seed,
        num_clients=num_clients,
        participation_rate=config.federation.participation_rate,
        available_clients=available_clients,
    )

    # ── Training loop (identical to train(), plus storage hook) ──
    for r in range(total_rounds):
        round_start = time.time()

        # 1. Hash global model BEFORE aggregation.
        global_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        hash_pre = hash_model(global_sd)

        # 2. Client selection.
        selection_seed = derive_seed(run_seed, f"client_selection_round_{r}")
        rng = random.Random(selection_seed)
        selected = sorted(rng.sample(available_clients, num_selected))

        # Determine if this round's updates should be stored.
        # Rounds are stored 1-indexed: Δt, 2Δt, ... (so r+1).
        store_this_round = (r + 1) % store_updates_every == 0
        if store_this_round:
            round_updates_dir = updates_dir / f"round_{r + 1:03d}"
            round_updates_dir.mkdir(parents=True, exist_ok=True)

        # 3. Local training for each selected client.
        client_updates: List[tuple[Dict[str, torch.Tensor], int]] = []
        samples_per_client: Dict[int, int] = {}

        for client_id in selected:
            client_dl_seed = derive_seed(
                run_seed, f"dataloader_round_{r}_client_{client_id}",
            )
            loader = _make_client_dataloader(
                dataset=train_dataset,
                indices=partition[client_id],
                batch_size=config.data.batch_size,
                generator_seed=client_dl_seed,
            )

            client_model = copy.deepcopy(model)

            local_train_seed = derive_seed(
                run_seed, f"local_train_round_{r}_client_{client_id}",
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

            # ── STORAGE HOOK (pure read, no side effects) ──
            # client_update = client_state − global_state_before_round.
            # Reads only; does not touch PRNG or model. Saved on CPU.
            if store_this_round:
                update = {
                    k: (updated_sd[k].float().cpu() - global_sd[k].float().cpu())
                    for k in updated_sd
                }
                torch.save(
                    update,
                    round_updates_dir / f"client_{client_id:02d}.pt",
                )

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
        if (r + 1) % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"round_{r + 1:03d}.pt"
            torch.save(aggregated_sd, ckpt_path)

        elapsed = time.time() - round_start
        store_tag = " [stored updates]" if store_this_round else ""
        logger.info(
            "Round %d/%d — acc: %.4f, loss: %.4f (%.1fs)%s",
            r + 1, total_rounds, test_acc, test_loss, elapsed, store_tag,
        )

    # ── Finalisation ─────────────────────────────────────────────
    final_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    final_path = run_dir / "final_model.pt"
    torch.save(final_sd, final_path)

    final_hash = hash_model(final_sd)
    logger.info("Final model saved: %s", final_path)
    logger.info("Final model hash: %s", final_hash)

    # ── Hash-preservation check ──────────────────────────────────
    if expected_final_hash is not None:
        if final_hash != expected_final_hash:
            raise AssertionError(
                f"HASH MISMATCH — the storage hook introduced a side "
                f"effect!\n  expected: {expected_final_hash}\n  "
                f"actual:   {final_hash}\n"
                f"Training is NOT bit-identical to the original. Stop "
                f"and debug before using these stored updates."
            )
        logger.info(
            "HASH-PRESERVATION CHECK PASSED — final model is "
            "bit-identical to the original training.",
        )

    # ── Participation log ────────────────────────────────────────
    log_path = run_dir / "participation_log.json"
    log.save(log_path)

    # ── Manifest ─────────────────────────────────────────────────
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

    logger.info("FedEraser prep complete. Output at: %s", run_dir)
    return run_dir
