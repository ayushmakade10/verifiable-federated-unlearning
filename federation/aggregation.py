"""
federation/aggregation.py — FedAvg Weighted Aggregation
========================================================

Implements the server-side aggregation step of FedAvg: a weighted
average of client model updates, where each client's weight is
proportional to its number of training samples.

    w_global = Σ (n_k / n_total) * w_k

This is the standard aggregation rule from McMahan et al. (2017).

Usage:
    from federation.aggregation import fed_avg
    new_sd = fed_avg(client_updates)
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch


def fed_avg(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """Aggregate client model updates via weighted averaging.

    Args:
        client_updates: A list of (state_dict, num_samples) tuples,
                        one per client that participated this round.
                        All state_dicts must have the same keys and
                        tensor shapes.

    Returns:
        A new state_dict representing the weighted average of all
        client models. Tensors are on CPU.

    Raises:
        ValueError: If client_updates is empty.
    """
    if not client_updates:
        raise ValueError("fed_avg requires at least one client update")

    # Total samples across all participating clients.
    total_samples = sum(n for _, n in client_updates)

    # Start from a zeroed copy of the first client's state_dict.
    avg_state = {}
    first_sd = client_updates[0][0]
    for key in first_sd:
        avg_state[key] = torch.zeros_like(first_sd[key], dtype=torch.float32)

    # Accumulate weighted contributions.
    for state_dict, num_samples in client_updates:
        weight = num_samples / total_samples
        for key in avg_state:
            avg_state[key] += weight * state_dict[key].float()

    # Cast back to original dtypes (BN running stats may be float32 already,
    # but integer counts like num_batches_tracked need to stay as-is).
    for key in avg_state:
        original_dtype = first_sd[key].dtype
        if original_dtype != avg_state[key].dtype:
            avg_state[key] = avg_state[key].to(original_dtype)

    return avg_state


# ── Verification ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("aggregation.py: running verification\n")

    # 1. Equal-weight averaging.
    sd1 = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.0])}
    sd2 = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([2.0])}
    result = fed_avg([(sd1, 100), (sd2, 100)])
    assert torch.allclose(result["w"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(result["b"], torch.tensor([1.0]))
    print("[1/3] Equal-weight averaging correct")

    # 2. Weighted averaging (client 1 has 3x the data).
    result = fed_avg([(sd1, 300), (sd2, 100)])
    expected_w = 0.75 * torch.tensor([1.0, 2.0]) + 0.25 * torch.tensor([3.0, 4.0])
    assert torch.allclose(result["w"], expected_w)
    print("[2/3] Weighted averaging correct")

    # 3. Single client.
    result = fed_avg([(sd1, 100)])
    assert torch.allclose(result["w"], sd1["w"])
    print("[3/3] Single-client pass-through correct")

    print("\naggregation.py: all checks passed ✓")
