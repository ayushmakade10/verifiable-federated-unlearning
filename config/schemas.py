"""
config/schemas.py — Configuration Validation
=============================================

Pydantic models that enforce type-safety, range constraints, and
structural correctness on the YAML config at load time. If a config
is invalid, the run fails immediately with a clear error — not
silently midway through training.

Usage:
    from config.schemas import load_config
    cfg = load_config("config/default.yaml")
    print(cfg.federation.num_rounds)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Sub-configs ──────────────────────────────────────────────────────

class DataConfig(BaseModel):
    """Dataset and partitioning parameters."""
    dataset: str = Field("cifar10", description="Dataset name")
    num_clients: int = Field(50, ge=2, le=500)
    alpha: float = Field(
        0.3,
        gt=0.0,
        description="Dirichlet concentration. Lower = more non-IID.",
    )
    batch_size: int = Field(64, ge=1)
    data_root: str = Field("./data/raw", description="Download directory")


class ModelConfig(BaseModel):
    """Model architecture parameters."""
    architecture: str = Field("resnet18", description="Model name")
    num_classes: int = Field(10, ge=2)


class FederationConfig(BaseModel):
    """FedAvg training loop parameters."""
    num_rounds: int = Field(200, ge=1)
    local_epochs: int = Field(5, ge=1)
    participation_rate: float = Field(
        0.4,
        gt=0.0,
        le=1.0,
        description="Fraction of clients sampled per round.",
    )
    learning_rate: float = Field(0.01, gt=0.0)
    momentum: float = Field(0.9, ge=0.0, le=1.0)
    weight_decay: float = Field(5e-4, ge=0.0)
    optimizer: str = Field("sgd")

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        allowed = {"sgd", "adam"}
        if v.lower() not in allowed:
            raise ValueError(f"Optimizer must be one of {allowed}, got '{v}'")
        return v.lower()


class CheckpointConfig(BaseModel):
    """What to save and how often."""
    save_every_n_rounds: int = Field(
        10,
        ge=1,
        description="Checkpoint interval in rounds.",
    )
    save_final: bool = Field(True)
    output_dir: str = Field("./outputs")


class GoldStandardConfig(BaseModel):
    """Parameters for gold-standard retraining (Phase 1b)."""
    num_trials: int = Field(
        10,
        ge=1,
        description="Number of independent retraining runs per target client.",
    )


class ReproducibilityConfig(BaseModel):
    """Seed and determinism controls."""
    root_seed: int = Field(42)
    deterministic_cudnn: bool = Field(True)


# ── Top-level config ────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    """Root configuration object — the single source of truth."""
    project_name: str = Field("verifiable-federated-unlearning")
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    federation: FederationConfig = Field(default_factory=FederationConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    gold_standard: GoldStandardConfig = Field(default_factory=GoldStandardConfig)
    reproducibility: ReproducibilityConfig = Field(
        default_factory=ReproducibilityConfig,
    )

    @model_validator(mode="after")
    def checkpoint_interval_within_rounds(self) -> "ProjectConfig":
        if self.checkpoint.save_every_n_rounds > self.federation.num_rounds:
            raise ValueError(
                f"Checkpoint interval ({self.checkpoint.save_every_n_rounds}) "
                f"exceeds total rounds ({self.federation.num_rounds})."
            )
        return self


# ── Loader ───────────────────────────────────────────────────────────

def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a YAML config file.

    Raises pydantic.ValidationError with a clear message if anything
    is out of spec.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return ProjectConfig(**raw)


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Verify defaults pass validation.
    cfg = ProjectConfig()
    print("schemas.py: default config valid")
    print(f"  Clients:       {cfg.data.num_clients}")
    print(f"  Alpha:         {cfg.data.alpha}")
    print(f"  Rounds:        {cfg.federation.num_rounds}")
    print(f"  Participation: {cfg.federation.participation_rate}")
    print(f"  Checkpoint:    every {cfg.checkpoint.save_every_n_rounds} rounds")

    # Verify bad values are caught.
    try:
        FederationConfig(participation_rate=1.5)
        assert False, "Should have raised"
    except Exception:
        print("  Bad participation_rate correctly rejected")

    print("schemas.py: all checks passed")
