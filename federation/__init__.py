"""
federation/ — FedAvg Training Pipeline
========================================

Provides the resumable training loop, local client training,
and weighted aggregation for Federated Averaging.
"""

from federation.trainer import train
from federation.client import train_local
from federation.aggregation import fed_avg

__all__ = ["train", "train_local", "fed_avg"]
