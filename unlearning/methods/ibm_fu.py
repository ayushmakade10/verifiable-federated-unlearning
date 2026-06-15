"""
unlearning/methods/ibm_fu.py — IBM Federated Unlearning
=========================================================

Three-stage unlearning method from IBM Research (Wu et al., 2022):

  1. **Reference model**: Approximate the global model without the
     target client's contribution via analytical subtraction.
  2. **PGD unlearning**: Projected Gradient Descent — gradient ascent
     on the target client's data, constrained to an L2 ball around
     the reference model.
  3. **Recovery**: Standard FedAvg rounds on the remaining clients
     (handled externally via the existing ``train()`` function).

This module implements Stages 1 and 2. Stage 3 is orchestrated by
the runner script (``run_phase4b_ibm_fu.py``) using the existing
``federation.trainer.train()`` function.

Reference:
    Wu, C. et al. (2022). "Federated Unlearning with Knowledge
    Distillation". https://github.com/IBM/federated-unlearning

Specification reference: Section 9.3, Method 2.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

logger = logging.getLogger(__name__)


# ── Stage 1: Reference Model ─────────────────────────────────────


def compute_reference_model(
    global_state_dict: Dict[str, torch.Tensor],
    client_state_dict: Dict[str, torch.Tensor],
    num_clients: int,
) -> Dict[str, torch.Tensor]:
    """Approximate the global model without the target client.

    Uses the FedAvg contribution removal formula:

        w_ref = (N / (N-1)) * w_global - (1 / (N-1)) * w_client

    This analytically reverses the target client's weighted
    contribution to the aggregated global model.

    Args:
        global_state_dict:  Final global model weights (round 200).
        client_state_dict:  Target client's local model weights,
                            reconstructed by running one pass of
                            ``train_local()`` on the global model.
        num_clients:        Total number of FL clients (e.g. 50).

    Returns:
        Reference model state_dict (all tensors on CPU).
    """
    n = num_clients
    scale_global = n / (n - 1)
    scale_client = 1.0 / (n - 1)

    w_ref: Dict[str, torch.Tensor] = {}
    for key in global_state_dict:
        w_ref[key] = (
            scale_global * global_state_dict[key].float()
            - scale_client * client_state_dict[key].float()
        )

    logger.info(
        "Reference model computed: N=%d, scale_global=%.4f, "
        "scale_client=%.4f",
        n, scale_global, scale_client,
    )
    return w_ref


# ── Ball Radius Computation ──────────────────────────────────────


def compute_ball_radius(
    reference_state_dict: Dict[str, torch.Tensor],
    model_builder: Callable[[], nn.Module],
    num_random_inits: int = 10,
    seed_offset: int = 9000,
) -> float:
    """Compute the L2-ball radius for PGD projection.

    Generates ``num_random_inits`` randomly initialised models,
    measures each one's L2 distance to the reference model, and
    returns ``mean_distance / 3``. This sets a constraint radius
    that is meaningful relative to the model's parameter space.

    Args:
        reference_state_dict: The w_ref state_dict (Stage 1 output).
        model_builder:        Callable that returns a fresh,
                              randomly-initialised model (e.g.
                              ``build_model``).
        num_random_inits:     Number of random models to average
                              over (default 10).
        seed_offset:          Base seed for random inits, chosen to
                              avoid collision with training seeds.

    Returns:
        The ball radius (float, > 0).
    """
    distances = []

    for i in range(num_random_inits):
        torch.manual_seed(seed_offset + i)
        random_model = model_builder()
        random_sd = random_model.state_dict()

        l2_sq = 0.0
        for key in reference_state_dict:
            diff = reference_state_dict[key].float() - random_sd[key].float()
            l2_sq += (diff ** 2).sum().item()
        l2_dist = l2_sq ** 0.5
        distances.append(l2_dist)

    mean_dist = sum(distances) / len(distances)
    ball_radius = mean_dist / 3.0

    logger.info(
        "Ball radius computation: %d random inits, "
        "mean_L2=%.4f, radius=%.4f",
        num_random_inits, mean_dist, ball_radius,
    )
    return ball_radius


# ── Stage 2: PGD Unlearning ──────────────────────────────────────


def _compute_l2_distance(
    model: nn.Module,
    reference_state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    """Compute L2 distance between model parameters and reference."""
    l2_sq = 0.0
    model_sd = model.state_dict()
    for key in reference_state_dict:
        diff = model_sd[key].float().to(device) - reference_state_dict[key].float().to(device)
        l2_sq += (diff ** 2).sum().item()
    return l2_sq ** 0.5


def _project_to_ball(
    model: nn.Module,
    reference_state_dict: Dict[str, torch.Tensor],
    ball_radius: float,
    device: torch.device,
) -> float:
    """Project model parameters back onto the L2 ball if outside.

    If the current model is within the ball, no change is made.
    Otherwise, the displacement from w_ref is normalised and
    scaled to ``ball_radius``.

    Returns:
        The L2 distance after projection (for logging).
    """
    # Collect displacement vectors.
    displacement: Dict[str, torch.Tensor] = {}
    l2_sq = 0.0

    model_sd = model.state_dict()
    for key in reference_state_dict:
        diff = model_sd[key].float().to(device) - reference_state_dict[key].float().to(device)
        displacement[key] = diff
        l2_sq += (diff ** 2).sum().item()

    l2_dist = l2_sq ** 0.5

    if l2_dist <= ball_radius:
        return l2_dist

    # Project: w = w_ref + direction * ball_radius
    scale = ball_radius / l2_dist
    new_sd = {}
    for key in reference_state_dict:
        new_sd[key] = (
            reference_state_dict[key].float().to(device)
            + displacement[key] * scale
        )

    model.load_state_dict(new_sd)
    return ball_radius


def run_pgd_unlearning(
    model_state_dict: Dict[str, torch.Tensor],
    reference_state_dict: Dict[str, torch.Tensor],
    ball_radius: float,
    dataloader: torch.utils.data.DataLoader,
    model_builder: Callable[[], nn.Module],
    num_epochs: int = 5,
    lr: float = 0.01,
    momentum: float = 0.9,
    clip_grad: float = 5.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Run Projected Gradient Descent (gradient ascent + L2 projection).

    Starting from the reference model (w_ref), this function:
      1. Runs gradient ascent (negated cross-entropy loss) on the
         target client's data.
      2. After each optimizer step, projects the model back onto the
         L2 ball of radius ``ball_radius`` centred at w_ref.

    Args:
        model_state_dict:     Starting weights (w_ref, same as
                              reference_state_dict in standard usage).
        reference_state_dict: Projection anchor (w_ref). Kept frozen
                              throughout PGD.
        ball_radius:          Maximum L2 distance from w_ref.
        dataloader:           DataLoader over the target client's data.
        model_builder:        Callable returning a fresh model instance
                              (used to instantiate architecture).
        num_epochs:           Number of PGD epochs (default 5).
        lr:                   SGD learning rate (default 0.01).
        momentum:             SGD momentum (default 0.9).
        clip_grad:            Max gradient norm for clipping (default 5.0).
        device:               Computation device (cpu or cuda).

    Returns:
        The unlearned model's state_dict with all tensors on CPU.
    """
    # Build model and load starting weights.
    model = model_builder()
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.train()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
    )
    criterion = nn.CrossEntropyLoss()

    num_samples = len(dataloader.dataset)
    logger.info(
        "Starting PGD unlearning: %d epochs, lr=%.4f, "
        "ball_radius=%.4f, clip_grad=%.1f, %d samples",
        num_epochs, lr, ball_radius, clip_grad, num_samples,
    )

    initial_dist = _compute_l2_distance(model, reference_state_dict, device)
    logger.info("  Initial L2 distance from w_ref: %.4f", initial_dist)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        num_projections = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Gradient ASCENT: negate loss before backward.
            negated_loss = -loss
            negated_loss.backward()

            # Gradient clipping.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad,
            )

            optimizer.step()

            # L2-ball projection.
            dist_before = _compute_l2_distance(
                model, reference_state_dict, device,
            )
            _project_to_ball(
                model, reference_state_dict, ball_radius, device,
            )
            if dist_before > ball_radius:
                num_projections += 1

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        final_dist = _compute_l2_distance(
            model, reference_state_dict, device,
        )
        logger.info(
            "  PGD epoch %d/%d — avg loss: %.4f, "
            "L2 from w_ref: %.4f, projections: %d/%d",
            epoch + 1, num_epochs, avg_loss,
            final_dist, num_projections, num_batches,
        )

    # Return weights on CPU.
    unlearned_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    logger.info("PGD unlearning complete.")
    return unlearned_sd
