"""
unlearning/methods/sga.py — Simple Stochastic Gradient Ascent
================================================================

Crude baseline unlearning: run gradient ascent on the target client's
data to maximise loss, pushing the model away from what it learned
from that client. No constraints, no projection, no knowledge
distillation — intentionally simple as the lowest-quality point on
the Phase 4b quality gradient.

Usage:
    from unlearning.methods.sga import run_sga

    unlearned_sd = run_sga(
        model_state_dict=original_sd,
        dataloader=client_0_loader,
        num_epochs=5,
        learning_rate=0.01,
        device=torch.device("cuda"),
    )

Specification reference: Section 9.3, Method 1.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from models.resnet import build_model

logger = logging.getLogger(__name__)


def run_sga(
    model_state_dict: Dict[str, torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    num_classes: int = 10,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> Dict[str, torch.Tensor]:
    """Run Stochastic Gradient Ascent on one client's data.

    Loads the given state_dict into a fresh ResNet-18, then trains
    with **negated loss** for ``num_epochs`` on the provided data.
    This maximises the cross-entropy loss on the target client's
    data, pushing the model to forget what it learned from that
    client.

    The optimizer mirrors the training setup (SGD with momentum and
    weight decay) to ensure the gradient ascent operates in the same
    parameter space as the original training.

    Args:
        model_state_dict: Trained model weights to start from.
        dataloader:       DataLoader over the target client's data.
        num_epochs:       Number of gradient ascent epochs.
        learning_rate:    Learning rate for gradient ascent.
        device:           Device for computation (cpu or cuda).
        num_classes:      Number of output classes (default 10 for CIFAR-10).
        momentum:         SGD momentum (default 0.9, matches training).
        weight_decay:     SGD weight decay (default 5e-4, matches training).

    Returns:
        The unlearned model's state_dict with all tensors on CPU.
    """
    # Build model and load weights.
    model = build_model(num_classes=num_classes)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.train()

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    num_samples = len(dataloader.dataset)
    logger.info(
        "Starting SGA: %d epochs, lr=%.4f, %d samples",
        num_epochs, learning_rate, num_samples,
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Gradient ASCENT: negate the loss before backward.
            # This maximises cross-entropy on the target client's data.
            negated_loss = -loss
            negated_loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(
            "  SGA epoch %d/%d — avg loss: %.4f (should increase)",
            epoch + 1, num_epochs, avg_loss,
        )

    # Return weights on CPU.
    unlearned_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    logger.info("SGA complete. Model state_dict returned.")
    return unlearned_sd
