"""
unlearning/methods/federaser.py — FedEraser
==============================================

Calibration-based unlearning from Liu et al. (2021, IWQoS).

FedEraser reconstructs the training trajectory by replaying stored
per-client updates WITHOUT the target client, calibrating each step:

  For each stored round t:
    1. Load stored per-client updates from round t.
    2. Drop the target client's update.
    3. For each remaining client c:
         a. Train c locally on the current calibrated model.
         b. cal_update_c = client_model − calibrated_model
         c. cos_sim = cos(stored_update_c, cal_update_c)  (whole-model)
         d. if cos_sim > 0:  use stored DIRECTION, fresh MAGNITUDE
            else:            fall back to the fresh calibration update
    4. Aggregate calibrated updates (FedAvg, weighted by samples).
    5. calibrated_model += aggregated_update.

The calibration step — cosine between stored and fresh updates, with
magnitude adjustment — is the method's core innovation: it reuses the
cheap stored directions but rescales them to the magnitudes a fresh
(target-free) model would produce.

Reference:
    Liu, G. et al. (2021). "FedEraser: Enabling Efficient Client-Level
    Data Removal from Federated Learning Models". IWQoS 2021.

Specification reference: Section 9.3, Method 4.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from config.schemas import ProjectConfig
from evidence.bundle import build_manifest, save_frozen_config_from_dict
from evidence.hashing import hash_model
from evidence.participation_log import ParticipationLog

# Reuse train()'s helpers for identical data/eval behaviour.
from federation.trainer import (
    _evaluate,
    _get_cifar10_transforms,
    _make_client_dataloader,
    _make_test_dataloader,
)
from models.resnet import build_model
from utils.seeding import derive_seed, set_global_seed

logger = logging.getLogger(__name__)


# ── Stored-Update Loading ────────────────────────────────────────


def load_stored_updates(
    prep_dir: Path,
    round_num: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Load all client updates stored at a given round.

    Reads ``prep_dir/client_updates/round_{round_num:03d}/client_*.pt``.

    Args:
        prep_dir:  The FedEraser prep run directory.
        round_num: The 1-indexed stored round (e.g. 5, 10, 20).

    Returns:
        Map ``{client_id: update_state_dict}``. Empty if the round
        directory does not exist.
    """
    round_dir = prep_dir / "client_updates" / f"round_{round_num:03d}"
    if not round_dir.exists():
        return {}

    updates: Dict[int, Dict[str, torch.Tensor]] = {}
    for ckpt_file in sorted(round_dir.glob("client_*.pt")):
        # Filename: client_NN.pt → client_id = NN.
        stem = ckpt_file.stem  # "client_NN"
        client_id = int(stem.split("_")[1])
        updates[client_id] = torch.load(ckpt_file, weights_only=True)

    return updates


# ── Flatten Utility (for whole-model cosine) ─────────────────────


def _flatten_update(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten an update state_dict into a single 1-D CPU vector."""
    return torch.cat([v.float().cpu().flatten() for v in update.values()])


# ── Calibration Training (E_cali epochs) ─────────────────────────


def _calibration_train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: ProjectConfig,
    device: torch.device,
    num_epochs: int,
) -> int:
    """Local training for exactly ``num_epochs`` epochs (CaliTrain).

    Mirrors ``federation.client.train_local`` but runs a custom
    (small) number of epochs — the calibration ratio r = E_cali /
    E_local that gives FedEraser its speed-up. The existing
    ``train_local`` is not modified.

    Trains ``model`` in place. Returns the number of samples seen.
    """
    model.train()
    fed_cfg = config.federation

    if fed_cfg.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=fed_cfg.learning_rate,
            momentum=fed_cfg.momentum,
            weight_decay=fed_cfg.weight_decay,
        )
    elif fed_cfg.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=fed_cfg.learning_rate,
            weight_decay=fed_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {fed_cfg.optimizer}")

    criterion = nn.CrossEntropyLoss()
    num_samples = len(dataloader.dataset)

    for _epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return num_samples


# ── Calibration ──────────────────────────────────────────────────


def calibrate_updates(
    stored_updates: Dict[int, Dict[str, torch.Tensor]],
    calibrated_model: nn.Module,
    participated_clients: List[int],
    samples_per_client: Dict[int, int],
    partition: Dict[int, List[int]],
    target_client: int,
    train_dataset,
    config: ProjectConfig,
    device: torch.device,
    round_seed_context: str,
    calibration_epochs: int,
) -> Dict[str, torch.Tensor]:
    """One FedEraser calibration step, faithful to the reference code.

    Implements ``unlearning_step_once`` from the authors' release
    (Liu et al. 2021). For each layer ℓ independently:

        oldU[ℓ] = mean_c(stored_client[ℓ]) − oldGM_t[ℓ]   (original)
        newU[ℓ] = mean_c(fresh_client[ℓ]) − newGM_t[ℓ]    (calibration)

        step_length[ℓ]    = ||oldU[ℓ]||                   (per-layer norm)
        step_direction[ℓ] = newU[ℓ] / ||newU[ℓ]||         (per-layer unit)

        newGM_{t+1}[ℓ] = newGM_t[ℓ] + step_length[ℓ] · step_direction[ℓ]

    Key points matching the reference:
      * Per-LAYER norms (not whole-model).
      * Magnitude from the AVERAGE stored client update (oldCM − oldGM_t).
        Our stored deltas already equal (client − oldGM_t), so the mean
        of stored deltas per layer IS oldU[ℓ] directly.
      * Direction from the AVERAGE fresh client model trained E_cali
        epochs on the current calibrated model.
      * Returns the COMPLETE new model state (not an additive delta).

    Args:
        stored_updates:       {client_id: stored delta state_dict}
                              (client − original_global at that round).
        calibrated_model:     Current calibrated model newGM_t (on device).
        participated_clients: Client IDs that participated at this round.
        samples_per_client:   {client_id: n_samples} (unused for the
                              per-layer mean, kept for interface parity).
        partition:            Full client partition.
        target_client:        Client to exclude.
        train_dataset:        CIFAR-10 training dataset.
        config:               Project configuration.
        device:               Computation device.
        round_seed_context:   Seed-derivation context for this round.
        calibration_epochs:   E_cali — local epochs for calibration.

    Returns:
        The complete new calibrated model state_dict (CPU).
    """
    retain_clients = [
        c for c in participated_clients
        if c != target_client and c in stored_updates
    ]
    if not retain_clients:
        raise ValueError(
            "No retain clients with stored updates this round."
        )

    newGM_t = {k: v.float().cpu() for k, v in calibrated_model.state_dict().items()}

    # ── oldU[ℓ] = mean over retain clients of stored deltas ──
    # Each stored delta already equals (client − original_global), so
    # the per-layer mean IS (mean(oldCM) − oldGM_t) for that round.
    old_update_mean: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(v) for k, v in newGM_t.items()
    }
    for c in retain_clients:
        for k in old_update_mean:
            old_update_mean[k] += stored_updates[c][k].float().cpu()
    for k in old_update_mean:
        old_update_mean[k] /= len(retain_clients)

    # ── Fresh calibration training: mean of fresh client models ──
    fresh_model_mean: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(v) for k, v in newGM_t.items()
    }
    for c in retain_clients:
        client_dl_seed = derive_seed(
            config.reproducibility.root_seed,
            f"{round_seed_context}_dataloader_client_{c}",
        )
        loader = _make_client_dataloader(
            dataset=train_dataset,
            indices=partition[c],
            batch_size=config.data.batch_size,
            generator_seed=client_dl_seed,
        )

        client_model = copy.deepcopy(calibrated_model)
        local_train_seed = derive_seed(
            config.reproducibility.root_seed,
            f"{round_seed_context}_localtrain_client_{c}",
        )
        set_global_seed(local_train_seed)

        _calibration_train(
            model=client_model,
            dataloader=loader,
            config=config,
            device=device,
            num_epochs=calibration_epochs,
        )
        client_sd = client_model.state_dict()
        for k in fresh_model_mean:
            fresh_model_mean[k] += client_sd[k].float().cpu()
    for k in fresh_model_mean:
        fresh_model_mean[k] /= len(retain_clients)

    # ── Per-layer step: newGM_t + ||oldU|| · (newU / ||newU||) ──
    # newU = mean(fresh_client) − newGM_t.
    return_state: Dict[str, torch.Tensor] = {}
    for k in newGM_t:
        new_update = fresh_model_mean[k] - newGM_t[k]

        step_length = torch.norm(old_update_mean[k])
        new_norm = torch.norm(new_update)

        if new_norm < 1e-12:
            # No fresh movement in this layer — keep it unchanged.
            return_state[k] = newGM_t[k].clone()
        else:
            step_direction = new_update / new_norm
            return_state[k] = newGM_t[k] + step_length * step_direction

    return return_state


# ── Main Orchestration ───────────────────────────────────────────


def _aggregate_stored_directly(
    stored_updates: Dict[int, Dict[str, torch.Tensor]],
    participated_clients: List[int],
    samples_per_client: Dict[int, int],
    target_client: int,
    init_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """First-round model: init + mean(stored retain deltas).

    The reference code's first reconstruction epoch is simply
    ``fedavg(retain_client_models)`` — the initial model has no
    target-client influence yet, so no calibration is needed. Our
    stored deltas satisfy ``delta = client − init_global``, so

        mean(client) = init_global + mean(delta)

    which is exactly the FedAvg of the retain client models. Returns
    the complete new model state (CPU). Target dropped.
    """
    retain = [
        c for c in participated_clients
        if c != target_client and c in stored_updates
    ]
    if not retain:
        raise ValueError("No stored updates to aggregate (first round).")

    return_state: Dict[str, torch.Tensor] = {}
    for key in init_state:
        delta_mean = torch.zeros_like(init_state[key].float())
        for c in retain:
            delta_mean += stored_updates[c][key].float().cpu()
        delta_mean /= len(retain)
        return_state[key] = init_state[key].float().cpu() + delta_mean
    return return_state


def run_federaser(
    prep_dir: Path,
    partition: Dict[int, List[int]],
    target_client: int,
    delta_t: int,
    config: ProjectConfig,
    run_id: str,
    run_seed: int,
    device: torch.device,
    checkpoint_every: int = 10,
    calibration_epochs: int = 3,
) -> Path:
    """Full FedEraser pipeline: calibrated trajectory reconstruction.

    Args:
        prep_dir:        FedEraser prep run directory (with stored updates).
        partition:       Full client partition (all 50 clients).
        target_client:   Client to unlearn (e.g. 0).
        delta_t:         Round interval — process stored rounds
                         [Δt, 2Δt, ..., 200]. Must be a multiple of the
                         prep run's storage interval (5).
        config:          Project configuration.
        run_id:          Output directory name.
        run_seed:        Seed for this FedEraser run's calibration.
        device:          Computation device.
        checkpoint_every: Save a checkpoint every N calibration rounds.
        calibration_epochs: E_cali — local epochs per calibration round.
                         With E_local=5, E_cali=3 gives r=0.6 (paper uses
                         r=0.5). The first round applies stored updates
                         directly with no calibration training.

    Returns:
        Path to the evidence bundle directory.
    """
    output_base = Path(config.checkpoint.output_dir)
    run_dir = output_base / run_id
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc).isoformat()

    # ── Read the original run_seed from the prep participation log ──
    # The calibration must start from the SAME initial model as the
    # original training, which used model_init derived from that seed.
    prep_log = ParticipationLog.load(
        str(prep_dir / "participation_log.json"),
    )
    original_run_seed = prep_log.run_seed

    logger.info(
        "FedEraser: run_id=%s, Δt=%d, target=%d, original_seed=%d, device=%s",
        run_id, delta_t, target_client, original_run_seed, device,
    )

    # ── Save frozen config ───────────────────────────────────────
    config_dict = config.model_dump()
    save_frozen_config_from_dict(config_dict, run_dir)

    # ── Reconstruct the SAME initial model as original training ──
    init_seed = derive_seed(original_run_seed, "model_init")
    set_global_seed(init_seed)
    calibrated_model = build_model(num_classes=config.model.num_classes)
    calibrated_model = calibrated_model.to(device)
    # Snapshot the initial state — needed for the first-round
    # reconstruction (init + mean stored deltas).
    init_state = {
        k: v.float().cpu() for k, v in calibrated_model.state_dict().items()
    }
    logger.info(
        "Initial model reconstructed (init_seed=%d, from original run)",
        init_seed,
    )

    # ── Dataset & test loader ────────────────────────────────────
    data_seed = derive_seed(run_seed, "federaser_data_loading")
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

    # ── Build participation log for the calibration trajectory ──
    retain_clients = sorted(
        [c for c in partition if c != target_client],
    )
    log = ParticipationLog(
        run_id=run_id,
        run_seed=run_seed,
        num_clients=len(retain_clients),
        participation_rate=config.federation.participation_rate,
        available_clients=retain_clients,
    )

    # ── Stored rounds to process: [Δt, 2Δt, ..., 200] ──
    total_rounds = config.federation.num_rounds
    stored_rounds = list(range(delta_t, total_rounds + 1, delta_t))
    logger.info(
        "Processing %d calibration rounds: %s%s",
        len(stored_rounds),
        stored_rounds[:5],
        "..." if len(stored_rounds) > 5 else "",
    )

    init_acc, init_loss = _evaluate(calibrated_model, test_loader, device)
    logger.info("Initial model: acc=%.4f, loss=%.4f", init_acc, init_loss)

    # ── Calibration loop ─────────────────────────────────────────
    for idx, t in enumerate(stored_rounds):
        round_start = time.time()

        global_sd = {k: v.cpu() for k, v in calibrated_model.state_dict().items()}
        hash_pre = hash_model(global_sd)

        # 1. Load stored updates for round t.
        stored_updates = load_stored_updates(prep_dir, t)
        if not stored_updates:
            raise FileNotFoundError(
                f"No stored updates found for round {t} in {prep_dir}. "
                f"Ensure the prep run stored at Δt dividing {delta_t}."
            )

        # 2. Which clients participated at round t (from prep log).
        #    Prep log is 0-indexed by round_id; stored round t maps to
        #    round_id = t - 1.
        prep_round_entry = prep_log.get_round(t - 1)
        participated = prep_round_entry["selected_clients"]
        samples_per_client = {
            int(k): v
            for k, v in prep_round_entry["num_samples_per_client"].items()
        }

        # 3-5. Reconstruct the new calibrated model state. Both paths
        # return a COMPLETE model state (not an additive delta).
        # First round: init + mean(stored retain deltas), no calibration
        # training — the initial model has no target-client influence
        # yet (reference round-0 is fedavg of retain client models).
        if idx == 0:
            new_model_state = _aggregate_stored_directly(
                stored_updates=stored_updates,
                participated_clients=participated,
                samples_per_client=samples_per_client,
                target_client=target_client,
                init_state=init_state,
            )
        else:
            new_model_state = calibrate_updates(
                stored_updates=stored_updates,
                calibrated_model=calibrated_model,
                participated_clients=participated,
                samples_per_client=samples_per_client,
                partition=partition,
                target_client=target_client,
                train_dataset=train_dataset,
                config=config,
                device=device,
                round_seed_context=f"federaser_{run_id}_round_{t}",
                calibration_epochs=calibration_epochs,
            )

        # 6. Apply: load the full reconstructed state, casting each
        # tensor back to the model's original dtype.
        model_sd = calibrated_model.state_dict()
        updated_sd = {
            k: new_model_state[k].to(model_sd[k].dtype)
            for k in model_sd
        }
        calibrated_model.load_state_dict(updated_sd)
        calibrated_model = calibrated_model.to(device)

        # 7. Evaluate + log.
        post_sd = {k: v.cpu() for k, v in calibrated_model.state_dict().items()}
        hash_post = hash_model(post_sd)
        test_acc, test_loss = _evaluate(calibrated_model, test_loader, device)

        retain_participated = [c for c in participated if c != target_client]
        retain_samples = {
            c: samples_per_client[c] for c in retain_participated
        }

        log.add_round(
            round_id=idx,
            selection_seed=prep_round_entry["selection_seed"],
            selected_clients=retain_participated,
            num_samples_per_client=retain_samples,
            global_model_hash_pre=hash_pre,
            global_model_hash_post=hash_post,
            test_accuracy=test_acc,
            test_loss=test_loss,
        )

        if (idx + 1) % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"round_{idx + 1:03d}.pt"
            torch.save(post_sd, ckpt_path)
            logger.info("  Checkpoint saved: %s", ckpt_path)

        elapsed = time.time() - round_start
        logger.info(
            "  Calibration round %d/%d (stored t=%d) — "
            "acc: %.4f, loss: %.4f (%.1fs)",
            idx + 1, len(stored_rounds), t,
            test_acc, test_loss, elapsed,
        )

    # ── Evidence bundle assembly ─────────────────────────────────
    logger.info("Assembling evidence bundle...")

    final_sd = {k: v.cpu() for k, v in calibrated_model.state_dict().items()}
    torch.save(final_sd, run_dir / "final_model.pt")

    log.save(run_dir / "participation_log.json")

    # Unlearning request.
    raw_dataset = torchvision.datasets.CIFAR10(
        root=config.data.data_root, train=True, download=False,
    )
    labels = np.array(raw_dataset.targets)
    client_labels = labels[partition[target_client]]
    counter = Counter(int(lbl) for lbl in client_labels)
    class_histogram = [
        counter.get(c, 0) for c in range(config.model.num_classes)
    ]
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

    end_time = datetime.now(timezone.utc).isoformat()
    build_manifest(
        run_dir=run_dir,
        run_id=run_id,
        run_seed=run_seed,
        total_rounds=len(stored_rounds),
        start_time=start_time,
        end_time=end_time,
    )

    logger.info("Evidence bundle saved to %s", run_dir)
    return run_dir
