"""
evidence/bundle.py — Evidence Bundle Assembly
===============================================

Utilities for assembling the evidence bundle directory layout
defined in Section 6.1 of the master document:

    outputs/{run_id}/
    ├── manifest.json
    ├── config.yaml
    ├── participation_log.json
    ├── unlearning_request.json   (not written by trainer — see below)
    ├── final_model.pt
    └── checkpoints/
        ├── round_010.pt
        └── ...

This module provides:
  - save_frozen_config()  — copy the training config into the run dir.
  - build_manifest()      — generate manifest.json with SHA-256 hashes
                            of every other file in the bundle.
  - create_unlearning_request() — utility for evaluation scenarios
                            (Phase 4+). Not called during training.

The trainer calls save_frozen_config() at the start of a run and
build_manifest() at the end. The unlearning request is created
externally when setting up evaluation scenarios.

Specification references: Sections 6.1, 6.4, 7.3.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from evidence.hashing import hash_file


def save_frozen_config(config_source: str | Path, run_dir: str | Path) -> Path:
    """Copy the training configuration into the run's output directory.

    This frozen copy ensures any result can be traced back to its
    exact settings, even if the source config is later modified.

    Args:
        config_source: Path to the original config YAML file.
        run_dir:       Path to the run's output directory.

    Returns:
        Path to the frozen config copy.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / "config.yaml"
    shutil.copy2(config_source, dest)
    return dest


def save_frozen_config_from_dict(
    config_dict: Dict[str, Any],
    run_dir: str | Path,
) -> Path:
    """Write a config dictionary as YAML into the run's output directory.

    Used when the config is already in memory (e.g. from a Pydantic
    model) rather than a file on disk.

    Args:
        config_dict: Configuration as a dictionary.
        run_dir:     Path to the run's output directory.

    Returns:
        Path to the frozen config file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / "config.yaml"
    with open(dest, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    return dest


def build_manifest(
    run_dir: str | Path,
    run_id: str,
    run_seed: int,
    total_rounds: int,
    dataset: str = "cifar10",
    architecture: str = "resnet18",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Path:
    """Generate manifest.json with SHA-256 hashes of every file in the bundle.

    The manifest is the integrity anchor for the evidence bundle.
    At High assurance, the auditor checks every hash in the manifest
    before proceeding with verification.

    Scans the run directory for:
      - config.yaml
      - participation_log.json
      - unlearning_request.json (if present)
      - final_model.pt
      - checkpoints/*.pt

    Args:
        run_dir:       Path to the run's output directory.
        run_id:        Identifier for this run.
        run_seed:      The seed used for this run.
        total_rounds:  Number of rounds completed.
        dataset:       Dataset identifier (default "cifar10").
        architecture:  Model architecture identifier (default "resnet18").
        start_time:    ISO 8601 timestamp of run start. Auto-set if None.
        end_time:      ISO 8601 timestamp of run end. Auto-set if None.

    Returns:
        Path to the generated manifest.json.
    """
    run_dir = Path(run_dir)

    now_iso = datetime.now(timezone.utc).isoformat()
    if start_time is None:
        start_time = now_iso
    if end_time is None:
        end_time = now_iso

    # Collect file hashes for every evidence artifact.
    file_hashes: Dict[str, str] = {}

    # Fixed files.
    for filename in ["config.yaml", "participation_log.json",
                     "unlearning_request.json", "final_model.pt"]:
        filepath = run_dir / filename
        if filepath.exists():
            file_hashes[filename] = hash_file(filepath)

    # Checkpoint files.
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        for ckpt_file in sorted(ckpt_dir.glob("*.pt")):
            relative = f"checkpoints/{ckpt_file.name}"
            file_hashes[relative] = hash_file(ckpt_file)

    manifest = {
        "run_id": run_id,
        "run_seed": run_seed,
        "start_time": start_time,
        "end_time": end_time,
        "total_rounds_completed": total_rounds,
        "dataset": dataset,
        "architecture": architecture,
        "file_hashes": file_hashes,
    }

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


# ── Unlearning Request (Phase 4+ utility) ────────────────────────

def create_unlearning_request(
    run_dir: str | Path,
    request_id: str,
    target_client_id: int,
    source_run_id: str,
    request_type: str = "client_deletion",
    timestamp: Optional[str] = None,
) -> Path:
    """Create an unlearning_request.json file in the evidence bundle.

    NOTE: This is NOT called during training. The unlearning request
    is an external artifact that the auditor receives as part of the
    evidence bundle. This utility exists for evaluation scenario setup
    in Phase 4+.

    The request object format is defined in Section 3.3:
        {
          "request_id": "req_001",
          "target_client_id": 7,
          "source_run_id": "run_001",
          "timestamp": "2026-05-10T14:30:00Z",
          "request_type": "client_deletion"
        }

    Args:
        run_dir:           Path to the run's output directory.
        request_id:        Unique identifier for this request.
        target_client_id:  The client requesting deletion.
        source_run_id:     The run from which to unlearn.
        request_type:      Type of unlearning request.
        timestamp:         ISO 8601 timestamp. Auto-set if None.

    Returns:
        Path to the created unlearning_request.json.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    request = {
        "request_id": request_id,
        "target_client_id": target_client_id,
        "source_run_id": source_run_id,
        "timestamp": timestamp,
        "request_type": request_type,
    }

    path = run_dir / "unlearning_request.json"
    with open(path, "w") as f:
        json.dump(request, f, indent=2)

    return path


# ── Verification ─────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("bundle.py: running verification\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir()

        # 1. Frozen config from dict.
        cfg = {"federation": {"num_rounds": 200, "lr": 0.01}}
        cfg_path = save_frozen_config_from_dict(cfg, run_dir)
        assert cfg_path.exists()
        print("[1/4] Frozen config saved")

        # 2. Create a dummy final model and participation log.
        import torch
        torch.save({"w": torch.randn(3)}, run_dir / "final_model.pt")
        with open(run_dir / "participation_log.json", "w") as f:
            json.dump({"rounds": []}, f)

        # 3. Create checkpoint.
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir()
        torch.save({"w": torch.randn(3)}, ckpt_dir / "round_010.pt")

        # 4. Build manifest.
        manifest_path = build_manifest(
            run_dir=run_dir,
            run_id="test",
            run_seed=42,
            total_rounds=10,
        )
        assert manifest_path.exists()
        with open(manifest_path) as f:
            m = json.load(f)
        assert "config.yaml" in m["file_hashes"]
        assert "final_model.pt" in m["file_hashes"]
        assert "participation_log.json" in m["file_hashes"]
        assert "checkpoints/round_010.pt" in m["file_hashes"]
        assert len(m["file_hashes"]) == 4
        print(f"[2/4] Manifest built with {len(m['file_hashes'])} file hashes")

        # 5. Unlearning request.
        req_path = create_unlearning_request(
            run_dir=run_dir,
            request_id="req_001",
            target_client_id=7,
            source_run_id="run_001",
        )
        assert req_path.exists()
        with open(req_path) as f:
            req = json.load(f)
        assert req["target_client_id"] == 7
        print("[3/4] Unlearning request created")

        # 6. Rebuild manifest — should now include unlearning_request.json.
        manifest_path = build_manifest(
            run_dir=run_dir, run_id="test", run_seed=42, total_rounds=10,
        )
        with open(manifest_path) as f:
            m = json.load(f)
        assert "unlearning_request.json" in m["file_hashes"]
        assert len(m["file_hashes"]) == 5
        print("[4/4] Manifest updated with unlearning request hash")

    print("\nbundle.py: all checks passed ✓")
