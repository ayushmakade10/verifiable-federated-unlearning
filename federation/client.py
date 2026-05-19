"""
federation/client.py — Local Client Training
==============================================

Implements the client-side training step of FedAvg. Each selected
client receives a copy of the global model, trains it on their
local data for a fixed number of epochs, and returns the updated
weights along with their sample count (used for weighted aggregation).

No global state is mutated — the function receives a model, trains
it, and returns the result. The caller (trainer.py) is responsible
for model copying and aggregation.

Usage:
    from federation.client import train_local
    updated_sd, n_samples = train_local(model, loader, config, device)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from config.schemas import ProjectConfig


def train_local(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: ProjectConfig,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Train a model on one client's data for local_epochs.

    Args:
        model:      The model to train (already a copy of the global model,
                    already on the correct device).
        dataloader: DataLoader over this client's subset of the training data.
        config:     Project configuration (provides lr, momentum, weight_decay,
                    local_epochs, optimizer type).
        device:     The device to train on (cpu or cuda).

    Returns:
        A tuple of:
          - state_dict: The trained model's state dictionary (on CPU).
          - num_samples: Total number of samples seen during training
            (len(dataloader.dataset)), used for FedAvg weighting.
    """
    model.train()

    # Build optimizer from config.
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

    for _epoch in range(fed_cfg.local_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Return weights on CPU to avoid holding GPU memory.
    return {k: v.cpu() for k, v in model.state_dict().items()}, num_samples
