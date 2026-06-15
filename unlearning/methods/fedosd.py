"""
unlearning/methods/fedosd.py — FedOSD: Federated Orthogonal Steepest Descent
===============================================================================

Server-side unlearning method from Pan et al. (2025):

  **Stage 1 — Unlearning rounds.** Each round, all 20 selected clients
  train locally (client 0 with an unlearning loss, others with standard
  CE). Per-client model updates (gradients) are collected. The server
  projects the forget gradient onto the subspace orthogonal to all
  retain gradients, then applies the orthogonal direction to the
  global model.

  **Stage 2 — Recovery rounds.** Retain clients (excluding client 0)
  train normally. Each gradient is modified to remove its reverting
  component (the projection onto the cumulative unlearning direction),
  then aggregated and applied with a small learning rate.

This is fundamentally different from SGA and IBM FU: the server
controls the update direction, not the client.

Reference:
    Pan, Z. et al. (2025). "Federated Orthogonal Steepest Descent".
    https://github.com/zibinpan/FedOSD

Specification reference: Section 9.3, Method 3.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from config.schemas import ProjectConfig
from evidence.bundle import build_manifest, save_frozen_config_from_dict
from evidence.hashing import hash_model
from evidence.participation_log import ParticipationLog
from federation.client import train_local
from models.resnet import build_model
from utils.seeding import derive_seed, seed_worker, set_global_seed

logger = logging.getLogger(__name__)


# ── Custom Loss ──────────────────────────────────────────────────


def unlearning_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """FedOSD's custom unlearning loss for the target client.

    .. math::

        \\mathcal{L} = -\\mathrm{mean}\\bigl(
            \\sum_c \\log(1 - p_c / 2) \\cdot \\mathbf{1}[y=c]
        \\bigr)

    where :math:`p = \\text{softmax}(\\text{logits})`.

    Minimising this loss pushes the predicted probability for the
    correct class toward zero — the model *forgets* the target
    client's data.

    Args:
        logits:  Raw model output [B, C].
        targets: Ground-truth class indices [B].

    Returns:
        Scalar loss tensor.
    """
    pred = torch.softmax(logits, dim=1)                            # [B, C]
    num_classes = pred.size(1)
    one_hot = F.one_hot(targets, num_classes=num_classes).float()  # [B, C]
    per_sample = (torch.log(1.0 - pred / 2.0) * one_hot).sum(1)   # [B]
    return -per_sample.mean()


# ── Flatten / Unflatten Utilities ────────────────────────────────


def _flatten_state_dict(
    sd: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Flatten all parameter tensors into a single 1-D CPU vector."""
    return torch.cat([v.float().cpu().flatten() for v in sd.values()])


def _unflatten_to_state_dict(
    flat: torch.Tensor,
    reference_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Reshape a flat vector back into a state_dict matching *reference_sd* shapes."""
    result: Dict[str, torch.Tensor] = {}
    offset = 0
    for key, ref in reference_sd.items():
        numel = ref.numel()
        result[key] = flat[offset:offset + numel].reshape(ref.shape)
        offset += numel
    return result


# ── Orthogonal Projection ───────────────────────────────────────


def compute_orthogonal_direction(
    retain_gradients: List[torch.Tensor],
    forget_gradient: torch.Tensor,
) -> torch.Tensor:
    """Compute the orthogonal steepest descent direction.

    Finds the component of ``forget_gradient`` orthogonal to the
    subspace spanned by all ``retain_gradients``, then scales it to
    match ``||forget_gradient||``.

    Computation (float64, memory-lean — no [K, P] stack)::

        for i:                                   # build Gram matrix
            v1[i]   = <a_i, gu>
            AAT[i,j] = <a_i, a_j>
        v2 = solve(AAT, v1)                      # [K]    — small solve
        proj = Σ_i v2[i] * a_i                   # [P]    — incremental
        d = gu - proj                            # [P]    — orthogonal component
        d = d * (||gu|| / ||d||)                 # scale to original norm

    The projection is computed in float64. At ResNet-18 scale, real FL
    retain gradients are highly correlated (all clients descend a
    similar loss landscape), so AAT is ill-conditioned. Float32 pinv
    on such a matrix leaves a large residual that breaks orthogonality
    (observed: dot ~5e4). Double precision plus a direct solve keeps
    the residual negligible. The Gram matrix and projection are built
    by iterating over gradients rather than stacking them, so peak
    memory stays near the float32 input size (~0.85 GB, not ~2.5 GB).

    Args:
        retain_gradients: List of K flat gradient vectors [P] (CPU).
        forget_gradient:  Single flat gradient vector [P] (CPU).

    Returns:
        Orthogonal direction d, flat vector [P] on CPU (float32).
    """
    # Compute the projection in float64 for numerical stability, but
    # WITHOUT stacking all gradients into a single [K, P] float64
    # matrix (which would transiently double memory). Instead we build
    # the [K, K] Gram matrix and the projection incrementally.
    gu = forget_gradient.cpu().double()
    gu_norm = gu.norm().item()
    K = len(retain_gradients)

    # Keep float64 views of each retain gradient (list, not stacked).
    A_rows = [g.cpu().double() for g in retain_gradients]  # K × [P]

    # v1[i] = <a_i, gu>;  AAT[i, j] = <a_i, a_j>.
    v1 = torch.empty(K, dtype=torch.float64)
    AAT = torch.empty((K, K), dtype=torch.float64)
    for i in range(K):
        v1[i] = torch.dot(A_rows[i], gu)
        for j in range(i, K):
            val = torch.dot(A_rows[i], A_rows[j])
            AAT[i, j] = val
            AAT[j, i] = val

    # Solve AAT @ v2 = v1 (more stable than explicit pinv).
    # Fall back to pseudoinverse only if AAT is singular (two retain
    # gradients exactly collinear — rare).
    try:
        v2 = torch.linalg.solve(AAT, v1)          # [K]
    except RuntimeError:
        logger.warning(
            "AAT is singular; falling back to pseudoinverse. "
            "Two retain gradients may be collinear.",
        )
        v2 = torch.linalg.pinv(AAT) @ v1          # [K]

    # proj = Σ_i v2[i] * a_i  — built incrementally (no [K, P] stack).
    proj = torch.zeros_like(gu)
    for i in range(K):
        proj += v2[i] * A_rows[i]

    # Orthogonal component.
    d = gu - proj

    # Scale to preserve ||gu||.
    d_norm = d.norm().item()
    if d_norm > 1e-10:
        d = d * (gu_norm / d_norm)
    else:
        logger.warning(
            "Orthogonal direction has near-zero norm (%.2e). "
            "Forget gradient may lie in the retain subspace.",
            d_norm,
        )

    logger.debug(
        "Orthogonal projection: K=%d, ||gu||=%.4f, "
        "||proj||=%.4f, ||d||=%.4f",
        K, gu_norm, proj.norm().item(), d.norm().item(),
    )

    # Return as float32 to match the rest of the pipeline.
    return d.float()


# ── Anti-Reverting ───────────────────────────────────────────────


def apply_anti_reverting(
    gradients: List[torch.Tensor],
    cumulative_unlearning_direction: torch.Tensor,
) -> List[torch.Tensor]:
    """Remove the reverting component from each gradient.

    For each gradient g::

        ga = cumulative_unlearning_direction
        reverting = (dot(g, ga) / ||ga||²) × ga
        g_mod = g - reverting
        g_mod = g_mod × (||g|| / ||g_mod||)        # preserve original norm

    This prevents recovery rounds from undoing the unlearning by
    removing the component of each gradient that points back toward
    the pre-unlearning model.

    Args:
        gradients:                        List of flat gradient vectors (CPU).
        cumulative_unlearning_direction:  ga = current_model − recovery_start (CPU).

    Returns:
        List of modified gradient vectors (CPU).
    """
    ga = cumulative_unlearning_direction.cpu().float()
    ga_norm_sq = ga.dot(ga).item()

    # If cumulative direction is near-zero (e.g. first recovery round
    # immediately after unlearning starts), skip modification.
    if ga_norm_sq < 1e-12:
        logger.debug(
            "Cumulative unlearning direction near-zero; "
            "skipping anti-reverting.",
        )
        return [g.clone() for g in gradients]

    modified = []
    for g in gradients:
        g = g.cpu().float()
        g_norm = g.norm().item()

        # Project out the reverting component.
        coeff = g.dot(ga).item() / ga_norm_sq
        g_mod = g - coeff * ga

        # Preserve original norm.
        g_mod_norm = g_mod.norm().item()
        if g_mod_norm > 1e-10:
            g_mod = g_mod * (g_norm / g_mod_norm)

        modified.append(g_mod)

    return modified


# ── Local Training with Pluggable Loss ───────────────────────────


def _train_local_with_loss(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    config: ProjectConfig,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Local training with a pluggable loss function.

    Identical to :func:`federation.client.train_local` except the
    loss function is parameterised. Used for client 0's unlearning
    loss during FedOSD unlearning rounds.

    No existing code is modified — this is a standalone copy with
    the loss function injected.

    Args:
        model:      Model to train (already a copy, already on device).
        dataloader: DataLoader over this client's data.
        loss_fn:    Loss function ``(logits, targets) → scalar``.
        config:     Project configuration.
        device:     Computation device.

    Returns:
        (state_dict on CPU, num_samples).
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

    num_samples = len(dataloader.dataset)

    for _epoch in range(fed_cfg.local_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

    return {k: v.cpu() for k, v in model.state_dict().items()}, num_samples


# ── Data Helpers ─────────────────────────────────────────────────


def _get_cifar10_transforms():
    """Standard CIFAR-10 transforms for ResNet-18."""
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
    """Create a reproducible DataLoader for one client's data."""
    subset = torch.utils.data.Subset(dataset, indices)
    g = torch.Generator()
    g.manual_seed(generator_seed)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a test DataLoader. Returns (accuracy, loss)."""
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


# ── Main Orchestration ───────────────────────────────────────────


def run_fedosd(
    global_state_dict: Dict[str, torch.Tensor],
    partition: Dict[int, List[int]],
    target_client: int,
    config: ProjectConfig,
    num_unlearning_rounds: int,
    num_recovery_rounds: int,
    unlearning_lr: float = 0.00008,
    recovery_lr: float = 0.0000002,
    run_id: str = "phase4b/fedosd_default",
    run_seed: int = 0,
    device: torch.device = torch.device("cpu"),
    checkpoint_every: int = 10,
) -> Path:
    """Full FedOSD pipeline: unlearning rounds followed by recovery rounds.

    This function implements a complete custom FL training loop for
    both stages, since neither can use the existing ``train()``
    function (unlearning needs orthogonal projection; recovery needs
    anti-reverting gradient modification).

    Args:
        global_state_dict:    Starting model weights (original round-200 model).
        partition:            Full client partition (all 50 clients).
        target_client:        Client to unlearn (e.g. 0).
        config:               Project configuration.
        num_unlearning_rounds: Number of unlearning rounds (Stage 1).
        num_recovery_rounds:  Number of recovery rounds (Stage 2).
        unlearning_lr:        Learning rate for unlearning updates (default 0.00008).
        recovery_lr:          Learning rate for recovery updates (default 0.0000002).
        run_id:               Output path under config.checkpoint.output_dir.
        run_seed:             Seed for all randomness in this run.
        device:               Computation device (GPU for training, CPU for projection).
        checkpoint_every:     Save checkpoints every N rounds (default 10).

    Returns:
        Path to the evidence bundle directory.
    """
    total_rounds = num_unlearning_rounds + num_recovery_rounds
    output_base = Path(config.checkpoint.output_dir)
    run_dir = output_base / run_id
    ckpt_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc).isoformat()

    logger.info(
        "FedOSD: %d unlearning + %d recovery rounds, "
        "unlearn_lr=%.6f, recover_lr=%.8f, device=%s",
        num_unlearning_rounds, num_recovery_rounds,
        unlearning_lr, recovery_lr, device,
    )

    # ── Save frozen config ───────────────────────────────────────
    config_dict = config.model_dump()
    save_frozen_config_from_dict(config_dict, run_dir)

    # ── Model setup ──────────────────────────────────────────────
    model = build_model(num_classes=config.model.num_classes)
    model.load_state_dict(global_state_dict)
    model = model.to(device)

    # ── Dataset ──────────────────────────────────────────────────
    data_seed = derive_seed(run_seed, "data_loading")
    set_global_seed(data_seed)

    train_transform, test_transform = _get_cifar10_transforms()
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data.data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data.data_root,
        train=False,
        download=True,
        transform=test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    # ── Client pools ─────────────────────────────────────────────
    retain_client_ids = sorted(
        [c for c in partition if c != target_client],
    )
    num_selected = max(
        1,
        round(len(partition) * config.federation.participation_rate),
    )

    logger.info(
        "Client pool: %d total, %d retain, %d selected/round, "
        "target=%d",
        len(partition), len(retain_client_ids),
        num_selected, target_client,
    )

    # ── Participation log ────────────────────────────────────────
    # Use num_clients=49 for consistency with SGA/IBM FU bundles
    # (the log represents the post-unlearning federation).
    log = ParticipationLog(
        run_id=run_id,
        run_seed=run_seed,
        num_clients=len(retain_client_ids),
        participation_rate=config.federation.participation_rate,
        available_clients=retain_client_ids,
    )

    # ── Initial evaluation ───────────────────────────────────────
    init_acc, init_loss = _evaluate(model, test_loader, device)
    logger.info(
        "Initial model: acc=%.4f, loss=%.4f", init_acc, init_loss,
    )

    # ==============================================================
    # Stage 1: Unlearning Rounds
    # ==============================================================
    logger.info("=" * 60)
    logger.info("Stage 1: Unlearning (%d rounds)", num_unlearning_rounds)
    logger.info("=" * 60)

    for r in range(num_unlearning_rounds):
        round_start = time.time()

        # 1. Snapshot global model.
        global_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        hash_pre = hash_model(global_sd)
        global_flat = _flatten_state_dict(global_sd)

        # 2. Client selection: force target + 19 random retain.
        selection_seed = derive_seed(run_seed, f"client_selection_round_{r}")
        rng = random.Random(selection_seed)
        selected_retain = sorted(
            rng.sample(retain_client_ids, num_selected - 1),
        )
        selected = sorted([target_client] + selected_retain)

        # 3. Local training + gradient collection.
        retain_grads: List[torch.Tensor] = []
        forget_grad: Optional[torch.Tensor] = None
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

            if client_id == target_client:
                client_sd, n_samples = _train_local_with_loss(
                    model=client_model,
                    dataloader=loader,
                    loss_fn=unlearning_ce_loss,
                    config=config,
                    device=device,
                )
            else:
                client_sd, n_samples = train_local(
                    model=client_model,
                    dataloader=loader,
                    config=config,
                    device=device,
                )

            # Gradient = client_after − global_before (on CPU).
            gradient = _flatten_state_dict(client_sd) - global_flat

            if client_id == target_client:
                forget_grad = gradient
            else:
                retain_grads.append(gradient)

            samples_per_client[client_id] = n_samples

        # 4. Orthogonal projection (on CPU).
        d = compute_orthogonal_direction(retain_grads, forget_grad)

        # 5. Validation: verify orthogonality (first round only).
        # Use the cosine (relative) rather than the raw dot product:
        # at ResNet-18 scale, gradient norms are large (~1e3), so a
        # raw dot product of even 1e3 can still be near-orthogonal.
        # The cosine normalises this out — true orthogonality means
        # |cos| ≈ 0 regardless of vector magnitude.
        if r == 0:
            d_norm = d.norm().item()
            max_cos = 0.0
            for i, rg in enumerate(retain_grads):
                rg_cpu = rg.cpu().float()
                rg_norm = rg_cpu.norm().item()
                dot_val = torch.dot(d, rg_cpu).item()
                cos_val = dot_val / (d_norm * rg_norm + 1e-12)
                max_cos = max(max_cos, abs(cos_val))
                if abs(cos_val) >= 1e-3:
                    raise AssertionError(
                        f"Orthogonality FAILED: d not orthogonal to "
                        f"retain gradient {i}: cos={cos_val:.6e} "
                        f"(dot={dot_val:.4e}, ||d||={d_norm:.2e}, "
                        f"||rg||={rg_norm:.2e})"
                    )
            logger.info(
                "Orthogonality validation PASSED "
                "(max |cos|=%.2e across %d retain gradients)",
                max_cos, len(retain_grads),
            )

        # 6. Update global model: w += unlearning_lr * d.
        d_sd = _unflatten_to_state_dict(d, global_sd)
        updated_sd: Dict[str, torch.Tensor] = {}
        for key in global_sd:
            updated_sd[key] = global_sd[key].float() + unlearning_lr * d_sd[key]
        model.load_state_dict(updated_sd)
        model = model.to(device)

        # 7. Post-round bookkeeping.
        post_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        hash_post = hash_model(post_sd)
        test_acc, test_loss = _evaluate(model, test_loader, device)

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

        # 8. Checkpoint.
        if (r + 1) % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"round_{r + 1:03d}.pt"
            torch.save(post_sd, ckpt_path)
            logger.info("  Checkpoint saved: %s", ckpt_path)

        elapsed = time.time() - round_start
        logger.info(
            "  Unlearning round %d/%d — acc: %.4f, loss: %.4f (%.1fs)",
            r + 1, num_unlearning_rounds, test_acc, test_loss, elapsed,
        )

    # ==============================================================
    # Stage 2: Recovery Rounds
    # ==============================================================
    logger.info("=" * 60)
    logger.info("Stage 2: Recovery (%d rounds)", num_recovery_rounds)
    logger.info("=" * 60)

    # Snapshot the model at the start of recovery for anti-reverting.
    recovery_start_sd = {
        k: v.cpu().clone() for k, v in model.state_dict().items()
    }
    recovery_start_flat = _flatten_state_dict(recovery_start_sd)

    for r in range(num_recovery_rounds):
        round_id = num_unlearning_rounds + r
        round_start = time.time()

        # 1. Snapshot global model.
        global_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        hash_pre = hash_model(global_sd)
        global_flat = _flatten_state_dict(global_sd)

        # 2. Client selection: 20 from retain clients only.
        selection_seed = derive_seed(
            run_seed, f"client_selection_round_{round_id}",
        )
        rng = random.Random(selection_seed)
        selected = sorted(rng.sample(retain_client_ids, num_selected))

        # 3. Local training + gradient collection.
        gradients: List[torch.Tensor] = []
        sample_counts: List[int] = []
        samples_per_client = {}

        for client_id in selected:
            client_dl_seed = derive_seed(
                run_seed,
                f"dataloader_round_{round_id}_client_{client_id}",
            )
            loader = _make_client_dataloader(
                dataset=train_dataset,
                indices=partition[client_id],
                batch_size=config.data.batch_size,
                generator_seed=client_dl_seed,
            )

            client_model = copy.deepcopy(model)
            local_train_seed = derive_seed(
                run_seed,
                f"local_train_round_{round_id}_client_{client_id}",
            )
            set_global_seed(local_train_seed)

            client_sd, n_samples = train_local(
                model=client_model,
                dataloader=loader,
                config=config,
                device=device,
            )

            gradient = _flatten_state_dict(client_sd) - global_flat
            gradients.append(gradient)
            sample_counts.append(n_samples)
            samples_per_client[client_id] = n_samples

        # 4. Cumulative unlearning direction.
        ga = global_flat - recovery_start_flat

        # 5. Anti-reverting modification.
        modified_grads = apply_anti_reverting(gradients, ga)

        # 6. Weighted aggregation of modified gradients.
        total_samples = sum(sample_counts)
        avg_grad = torch.zeros_like(modified_grads[0])
        for n, g_mod in zip(sample_counts, modified_grads):
            avg_grad += (n / total_samples) * g_mod

        # 7. Update global model: w += recovery_lr * avg_grad.
        update_sd = _unflatten_to_state_dict(avg_grad, global_sd)
        updated_sd = {}
        for key in global_sd:
            updated_sd[key] = global_sd[key].float() + recovery_lr * update_sd[key]
        model.load_state_dict(updated_sd)
        model = model.to(device)

        # 8. Post-round bookkeeping.
        post_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        hash_post = hash_model(post_sd)
        test_acc, test_loss = _evaluate(model, test_loader, device)

        log.add_round(
            round_id=round_id,
            selection_seed=selection_seed,
            selected_clients=selected,
            num_samples_per_client=samples_per_client,
            global_model_hash_pre=hash_pre,
            global_model_hash_post=hash_post,
            test_accuracy=test_acc,
            test_loss=test_loss,
        )

        # 9. Checkpoint.
        if (round_id + 1) % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"round_{round_id + 1:03d}.pt"
            torch.save(post_sd, ckpt_path)
            logger.info("  Checkpoint saved: %s", ckpt_path)

        elapsed = time.time() - round_start
        logger.info(
            "  Recovery round %d/%d (global %d/%d) — "
            "acc: %.4f, loss: %.4f (%.1fs)",
            r + 1, num_recovery_rounds,
            round_id + 1, total_rounds,
            test_acc, test_loss, elapsed,
        )

    # ==============================================================
    # Evidence Bundle Assembly
    # ==============================================================
    logger.info("Assembling evidence bundle...")

    # Final model.
    final_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(final_sd, run_dir / "final_model.pt")

    # Participation log.
    log.save(run_dir / "participation_log.json")

    # Unlearning request.
    raw_dataset = torchvision.datasets.CIFAR10(
        root=config.data.data_root, train=True, download=False,
    )
    labels = np.array(raw_dataset.targets)
    client_labels = labels[partition[target_client]]
    counter = Counter(int(lbl) for lbl in client_labels)
    class_histogram = [counter.get(c, 0) for c in range(config.model.num_classes)]

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

    # Manifest.
    end_time = datetime.now(timezone.utc).isoformat()
    build_manifest(
        run_dir=run_dir,
        run_id=run_id,
        run_seed=run_seed,
        total_rounds=total_rounds,
        start_time=start_time,
        end_time=end_time,
    )

    logger.info("Evidence bundle saved to %s", run_dir)
    return run_dir
