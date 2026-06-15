"""
experiments/failure_cases.py — Failure Case Generation Helpers
================================================================

Stateless library of functions for generating the 17 failure-case
evidence bundles defined in Phase 4a (Section 9.2).  Each function
takes explicit paths and parameters, produces a complete evidence
bundle that the existing verification pipeline can consume directly,
and returns the path to the bundle directory.

Two categories:

  Training-based (Experiments 2, 3, 4):
    Call ``federation.trainer.train()`` with failure-specific parameters.
    Produce genuine training output — model, checkpoints, log, manifest.

  File-manipulation (Experiments 5, 6):
    Copy an existing evidence bundle and tamper individual files.
    No training — only ``shutil`` and JSON editing.

Usage:
    from experiments.failure_cases import (
        generate_partial_retraining,
        generate_finetune_masquerade,
        generate_wrong_client_deletion,
        generate_rollback,
        generate_model_swap,
        find_wrong_clients,
    )
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from config.schemas import ProjectConfig
from evidence.bundle import build_manifest
from evidence.hashing import hash_file
from federation.trainer import train

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Utility: Wrong-client identification (Experiment 4)
# ═══════════════════════════════════════════════════════════════


def compute_class_histogram(
    indices: List[int],
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute the class distribution for a set of sample indices.

    Args:
        indices:     Sample indices belonging to this client.
        labels:      Full dataset label array.
        num_classes: Number of classes in the dataset.

    Returns:
        1-D array of shape (num_classes,) with per-class counts.
    """
    client_labels = labels[indices]
    counts = np.bincount(client_labels, minlength=num_classes)
    return counts.astype(np.float64)


def find_wrong_clients(
    partition: Dict[int, List[int]],
    target_client: int,
    labels: np.ndarray,
    num_classes: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Identify the most-different and most-similar clients to the target.

    Computes cosine similarity between the target client's class
    histogram and every other client's histogram.  Returns the two
    extremes — the client whose data distribution is most unlike the
    target (hardest to confuse) and the one most similar (easiest to
    confuse but wrong class signature).

    Args:
        partition:     Full client-to-indices mapping.
        target_client: The client requesting deletion (client 0).
        labels:        Full dataset label array.
        num_classes:   Number of dataset classes.

    Returns:
        Dict with keys ``"most_different"`` and ``"most_similar"``,
        each mapping to ``{"client_id": int, "cosine_similarity": float,
        "histogram": list}``.
    """
    target_hist = compute_class_histogram(
        partition[target_client], labels, num_classes,
    )
    target_norm = np.linalg.norm(target_hist)
    if target_norm == 0:
        raise ValueError(f"Target client {target_client} has no samples")

    similarities: List[tuple[int, float, np.ndarray]] = []
    for cid, indices in sorted(partition.items()):
        if cid == target_client:
            continue
        hist = compute_class_histogram(indices, labels, num_classes)
        norm = np.linalg.norm(hist)
        if norm == 0:
            continue
        cos_sim = float(np.dot(target_hist, hist) / (target_norm * norm))
        similarities.append((cid, cos_sim, hist))

    similarities.sort(key=lambda x: x[1])
    most_different = similarities[0]
    most_similar = similarities[-1]

    result = {
        "most_different": {
            "client_id": most_different[0],
            "cosine_similarity": most_different[1],
            "histogram": most_different[2].astype(int).tolist(),
        },
        "most_similar": {
            "client_id": most_similar[0],
            "cosine_similarity": most_similar[1],
            "histogram": most_similar[2].astype(int).tolist(),
        },
        "target_histogram": target_hist.astype(int).tolist(),
    }

    logger.info(
        "Wrong-client identification: most_different=client %d (cos=%.4f), "
        "most_similar=client %d (cos=%.4f)",
        most_different[0], most_different[1],
        most_similar[0], most_similar[1],
    )
    return result


# ═══════════════════════════════════════════════════════════════
#  Utility: Copy unlearning request into a bundle
# ═══════════════════════════════════════════════════════════════


def _copy_unlearning_request(
    source_bundle: Path,
    dest_bundle: Path,
) -> None:
    """Copy unlearning_request.json from source to dest if it exists."""
    src = source_bundle / "unlearning_request.json"
    if src.exists():
        shutil.copy2(src, dest_bundle / "unlearning_request.json")


def _ensure_unlearning_request(
    bundle_dir: Path,
    source_bundle: Path,
) -> None:
    """Ensure the bundle has an unlearning_request.json.

    Copies from ``source_bundle`` if the file does not already exist
    in ``bundle_dir``.
    """
    dest = bundle_dir / "unlearning_request.json"
    if not dest.exists():
        _copy_unlearning_request(source_bundle, bundle_dir)


# ═══════════════════════════════════════════════════════════════
#  Utility: Config deep copy with LR override (Experiment 3)
# ═══════════════════════════════════════════════════════════════


def _config_with_lr(config: ProjectConfig, lr: float) -> ProjectConfig:
    """Return a deep copy of the config with a modified learning rate.

    Uses Pydantic v2's ``model_copy(update=...)`` for a clean deep copy.
    The ``checkpoint_interval_within_rounds`` validator is satisfied
    because we do not change ``num_rounds`` in the config — the
    ``total_rounds`` parameter to ``train()`` handles that instead.
    """
    new_fed = config.federation.model_copy(
        update={"learning_rate": lr},
    )
    return config.model_copy(update={"federation": new_fed})


# ═══════════════════════════════════════════════════════════════
#  Experiment 2: Partial Retraining
# ═══════════════════════════════════════════════════════════════


def generate_partial_retraining(
    config: ProjectConfig,
    partition_minus_target: Dict[int, List[int]],
    run_seed: int,
    resume_round: int,
    checkpoint_path: Path,
    original_log_path: Path,
    run_id: str,
    source_bundle: Path,
) -> Path:
    """Generate a partial-retraining failure bundle (Experiment 2).

    Resumes training from an original-run checkpoint at round K and
    retrains the remaining rounds without the target client.  Uses
    the original run's seed so that round numbering and client
    selection are consistent with the original training.

    Args:
        config:                 Project configuration.
        partition_minus_target: Partition with target client removed.
        run_seed:               Original run seed (NOT a gold seed).
        resume_round:           Round to resume from (K).
        checkpoint_path:        Path to the checkpoint at round K.
        original_log_path:      Path to the original run's participation log.
        run_id:                 Output run ID (e.g. "phase4a/exp2_partial_K050").
        source_bundle:          Path to original bundle (for unlearning request).

    Returns:
        Path to the generated evidence bundle directory.
    """
    logger.info(
        "Experiment 2: partial retraining from round %d (run_id=%s)",
        resume_round, run_id,
    )

    # Load checkpoint state_dict.
    initial_model = torch.load(checkpoint_path, weights_only=True)

    # Load original participation log and extract entries before resume point.
    with open(original_log_path, "r", encoding="utf-8") as f:
        original_log_data = json.load(f)
    initial_log = [
        entry for entry in original_log_data["rounds"]
        if entry["round_id"] < resume_round
    ]
    logger.info(
        "  Loaded %d initial log entries (rounds 0–%d)",
        len(initial_log), resume_round - 1,
    )

    # Run training from the checkpoint.
    bundle_path = train(
        config=config,
        partition=partition_minus_target,
        run_seed=run_seed,
        start_round=resume_round,
        total_rounds=config.federation.num_rounds,
        initial_model=initial_model,
        initial_log=initial_log,
        run_id=run_id,
        save_checkpoints=True,
    )

    _ensure_unlearning_request(bundle_path, source_bundle)
    logger.info("  Bundle generated at %s", bundle_path)
    return bundle_path


# ═══════════════════════════════════════════════════════════════
#  Experiment 3: Fine-Tuning Masquerade
# ═══════════════════════════════════════════════════════════════


def generate_finetune_masquerade(
    config: ProjectConfig,
    partition_minus_target: Dict[int, List[int]],
    run_seed: int,
    original_model_path: Path,
    num_rounds: int,
    learning_rate: float,
    run_id: str,
    source_bundle: Path,
) -> Path:
    """Generate a fine-tuning masquerade failure bundle (Experiment 3).

    Starts from the original final model and fine-tunes it briefly on
    the remaining clients' data.  The provider is attempting to pass
    off a minor fine-tune as genuine retraining.

    Args:
        config:                 Project configuration.
        partition_minus_target: Partition with target client removed.
        run_seed:               Seed for this fine-tuning run (derived).
        original_model_path:    Path to the original final model.
        num_rounds:             Number of fine-tuning rounds (10 or 20).
        learning_rate:          LR for fine-tuning (0.01 or 0.001).
        run_id:                 Output run ID.
        source_bundle:          Path to original bundle (for unlearning request).

    Returns:
        Path to the generated evidence bundle directory.
    """
    logger.info(
        "Experiment 3: fine-tuning masquerade (%d rounds, lr=%.4f, "
        "run_id=%s)",
        num_rounds, learning_rate, run_id,
    )

    initial_model = torch.load(original_model_path, weights_only=True)
    modified_config = _config_with_lr(config, learning_rate)

    bundle_path = train(
        config=modified_config,
        partition=partition_minus_target,
        run_seed=run_seed,
        start_round=0,
        total_rounds=num_rounds,
        initial_model=initial_model,
        run_id=run_id,
        save_checkpoints=True,
        checkpoint_every=10,
    )

    _ensure_unlearning_request(bundle_path, source_bundle)
    logger.info("  Bundle generated at %s", bundle_path)
    return bundle_path


# ═══════════════════════════════════════════════════════════════
#  Experiment 4: Wrong Client Deletion
# ═══════════════════════════════════════════════════════════════


def generate_wrong_client_deletion(
    config: ProjectConfig,
    full_partition: Dict[int, List[int]],
    run_seed: int,
    wrong_client_id: int,
    run_id: str,
    source_bundle: Path,
) -> Path:
    """Generate a wrong-client-deletion failure bundle (Experiment 4).

    Performs a full 200-round retraining excluding the wrong client
    (instead of the actual target client).  The partition still
    includes the real target, so the model retains its influence.

    Args:
        config:          Project configuration.
        full_partition:  Full partition (all 50 clients).
        run_seed:        Seed for this wrong-deletion run (derived from
                         root seed and wrong client ID).
        wrong_client_id: The client incorrectly removed.
        run_id:          Output run ID.
        source_bundle:   Path to original bundle (for unlearning request).

    Returns:
        Path to the generated evidence bundle directory.
    """
    logger.info(
        "Experiment 4: wrong client deletion (removing client %d, "
        "run_id=%s)",
        wrong_client_id, run_id,
    )

    # Remove the wrong client from the partition.
    partition_minus_wrong = {
        k: v for k, v in full_partition.items() if k != wrong_client_id
    }

    bundle_path = train(
        config=config,
        partition=partition_minus_wrong,
        run_seed=run_seed,
        run_id=run_id,
        save_checkpoints=True,
    )

    _ensure_unlearning_request(bundle_path, source_bundle)
    logger.info("  Bundle generated at %s", bundle_path)
    return bundle_path


# ═══════════════════════════════════════════════════════════════
#  Experiment 5: Rollback-Only
# ═══════════════════════════════════════════════════════════════


def generate_rollback(
    source_bundle: Path,
    checkpoint_round: int,
    update_manifest: bool,
    run_id: str,
    output_base: Path,
) -> Path:
    """Generate a rollback-only failure bundle (Experiment 5).

    Copies the original run's entire evidence bundle, then replaces
    ``final_model.pt`` with the checkpoint from the specified round.

    Two manifest variants:
      - ``update_manifest=True``: Rebuild manifest so hashes are
        consistent. Tests whether *behavioral* checks detect the
        rollback (Check 5 integrity will pass).
      - ``update_manifest=False``: Leave the original manifest intact.
        The hash for ``final_model.pt`` will mismatch, so Check 5
        High catches it (tests integrity detection).

    Args:
        source_bundle:    Path to the original run's evidence bundle.
        checkpoint_round: Which checkpoint to use as "final" model
                          (e.g. 50 or 150).
        update_manifest:  Whether to rebuild the manifest.
        run_id:           Output run ID.
        output_base:      Base output directory (e.g. ``./outputs``).

    Returns:
        Path to the generated evidence bundle directory.
    """
    variant = "updated" if update_manifest else "stale"
    logger.info(
        "Experiment 5: rollback to round %d (%s manifest, run_id=%s)",
        checkpoint_round, variant, run_id,
    )

    dest = output_base / run_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_bundle, dest)

    # Swap the final model with the checkpoint.
    ckpt_path = dest / "checkpoints" / f"round_{checkpoint_round:03d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path} "
            f"(needed for rollback to round {checkpoint_round})"
        )

    final_model_path = dest / "final_model.pt"
    shutil.copy2(ckpt_path, final_model_path)
    logger.info("  Replaced final_model.pt with checkpoint round %d", checkpoint_round)

    if update_manifest:
        _rebuild_manifest_from_existing(dest)
        logger.info("  Manifest rebuilt with updated hashes")

    logger.info("  Bundle generated at %s", dest)
    return dest


# ═══════════════════════════════════════════════════════════════
#  Experiment 6: Inconsistent Evidence
# ═══════════════════════════════════════════════════════════════


def generate_model_swap(
    source_bundle: Path,
    swap_model_path: Path,
    run_id: str,
    output_base: Path,
) -> Path:
    """Experiment 6a: swap final_model.pt, keep original manifest.

    The manifest still contains the hash of the original final model,
    so Check 5 High should catch the mismatch.

    Args:
        source_bundle:   Path to the original run's evidence bundle.
        swap_model_path: Path to the model to swap in (e.g. a gold
                         trial's final_model.pt).
        run_id:          Output run ID.
        output_base:     Base output directory.

    Returns:
        Path to the generated bundle.
    """
    logger.info("Experiment 6a: model swap (run_id=%s)", run_id)

    dest = output_base / run_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_bundle, dest)

    shutil.copy2(swap_model_path, dest / "final_model.pt")
    logger.info("  Swapped final_model.pt with %s", swap_model_path)
    logger.info("  Bundle generated at %s", dest)
    return dest


def generate_log_edit(
    source_bundle: Path,
    target_round: int,
    run_id: str,
    output_base: Path,
) -> Path:
    """Experiment 6b: modify one round's selected_clients in the log.

    Changes the first client in the selected list for the target round
    to a different client.  This breaks seed verification (Check 5
    Basic) because the logged selection no longer matches what the
    seed would produce.

    Args:
        source_bundle: Path to the original run's evidence bundle.
        target_round:  Round to tamper (0-indexed).
        run_id:        Output run ID.
        output_base:   Base output directory.

    Returns:
        Path to the generated bundle.
    """
    logger.info(
        "Experiment 6b: log edit at round %d (run_id=%s)",
        target_round, run_id,
    )

    dest = output_base / run_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_bundle, dest)

    # Load and tamper the participation log.
    log_path = dest / "participation_log.json"
    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    # Find the target round entry and modify selected_clients.
    for entry in log_data["rounds"]:
        if entry["round_id"] == target_round:
            original_clients = entry["selected_clients"]
            # Swap the first selected client with a non-selected client.
            all_possible = set(range(log_data["num_clients"]))
            available = all_possible - set(original_clients)
            if available:
                replacement = min(available)  # Deterministic choice.
                tampered = [replacement] + original_clients[1:]
                entry["selected_clients"] = sorted(tampered)
                logger.info(
                    "  Round %d: changed client %d → %d in selection",
                    target_round, original_clients[0], replacement,
                )
            break

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    logger.info("  Bundle generated at %s", dest)
    return dest


def generate_hash_break(
    source_bundle: Path,
    target_round: int,
    run_id: str,
    output_base: Path,
) -> Path:
    """Experiment 6c: break the hash chain at a specific round.

    Modifies ``global_model_hash_post`` for the target round to a
    dummy value, breaking the chain link to round+1's pre-hash.

    Args:
        source_bundle: Path to the original run's evidence bundle.
        target_round:  Round whose post-hash to corrupt.
        run_id:        Output run ID.
        output_base:   Base output directory.

    Returns:
        Path to the generated bundle.
    """
    logger.info(
        "Experiment 6c: hash chain break at round %d (run_id=%s)",
        target_round, run_id,
    )

    dest = output_base / run_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_bundle, dest)

    log_path = dest / "participation_log.json"
    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    for entry in log_data["rounds"]:
        if entry["round_id"] == target_round:
            original_hash = entry["global_model_hash_post"]
            # Flip the first 8 characters to create an obviously wrong hash.
            tampered_hash = "deadbeef" + original_hash[8:]
            entry["global_model_hash_post"] = tampered_hash
            logger.info(
                "  Round %d: corrupted post-hash %s... → %s...",
                target_round, original_hash[:12], tampered_hash[:12],
            )
            break

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    logger.info("  Bundle generated at %s", dest)
    return dest


def generate_checkpoint_swap(
    source_bundle: Path,
    target_round: int,
    swap_checkpoint_path: Path,
    run_id: str,
    output_base: Path,
) -> Path:
    """Experiment 6d: replace a checkpoint with one from a different trial.

    Swaps ``checkpoints/round_{target_round}.pt`` with a checkpoint
    from a different training run.  The manifest still has the
    original hash, so Check 5 High should catch it.

    Args:
        source_bundle:       Path to the original run's evidence bundle.
        target_round:        Checkpoint round to replace (e.g. 100).
        swap_checkpoint_path: Path to the replacement checkpoint file.
        run_id:              Output run ID.
        output_base:         Base output directory.

    Returns:
        Path to the generated bundle.
    """
    logger.info(
        "Experiment 6d: checkpoint swap at round %d (run_id=%s)",
        target_round, run_id,
    )

    dest = output_base / run_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_bundle, dest)

    ckpt_dest = dest / "checkpoints" / f"round_{target_round:03d}.pt"
    shutil.copy2(swap_checkpoint_path, ckpt_dest)
    logger.info(
        "  Replaced checkpoint round %d with %s",
        target_round, swap_checkpoint_path,
    )

    logger.info("  Bundle generated at %s", dest)
    return dest


def generate_checkpoint_delete(
    source_bundle: Path,
    target_round: int,
    run_id: str,
    output_base: Path,
) -> Path:
    """Experiment 6e: delete a checkpoint file from the bundle.

    Removes ``checkpoints/round_{target_round}.pt``.  At Strong
    assurance, Check 5 verifies checkpoint existence at expected
    intervals, so this should be caught.

    Args:
        source_bundle: Path to the original run's evidence bundle.
        target_round:  Checkpoint round to delete (e.g. 100).
        run_id:        Output run ID.
        output_base:   Base output directory.

    Returns:
        Path to the generated bundle.
    """
    logger.info(
        "Experiment 6e: checkpoint deletion at round %d (run_id=%s)",
        target_round, run_id,
    )

    dest = output_base / run_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_bundle, dest)

    ckpt_path = dest / "checkpoints" / f"round_{target_round:03d}.pt"
    if ckpt_path.exists():
        ckpt_path.unlink()
        logger.info("  Deleted checkpoint round %d", target_round)
    else:
        logger.warning(
            "  Checkpoint round %d not found in source bundle", target_round,
        )

    logger.info("  Bundle generated at %s", dest)
    return dest


def generate_manifest_alter(
    source_bundle: Path,
    swap_model_path: Path,
    run_id: str,
    output_base: Path,
) -> Path:
    """Experiment 6f: swap model AND update manifest hash to match.

    Replaces ``final_model.pt`` with a different model and updates
    the manifest's hash for ``final_model.pt`` to match the new file.
    This makes the manifest internally consistent, but the
    participation log's hash chain still records the original model's
    ``global_model_hash_post`` for the final round, which will not
    match.  This tests whether cascading integrity constraints in the
    hash chain catch the swap even when the manifest is "fixed".

    Args:
        source_bundle:   Path to the original run's evidence bundle.
        swap_model_path: Path to the replacement model.
        run_id:          Output run ID.
        output_base:     Base output directory.

    Returns:
        Path to the generated bundle.
    """
    logger.info("Experiment 6f: manifest alter (run_id=%s)", run_id)

    dest = output_base / run_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_bundle, dest)

    # Swap the model.
    final_model_dest = dest / "final_model.pt"
    shutil.copy2(swap_model_path, final_model_dest)
    logger.info("  Swapped final_model.pt with %s", swap_model_path)

    # Update the manifest hash to match the new model file.
    manifest_path = dest / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    new_hash = hash_file(final_model_dest)
    old_hash = manifest["file_hashes"].get("final_model.pt", "N/A")
    manifest["file_hashes"]["final_model.pt"] = new_hash
    logger.info(
        "  Updated manifest: final_model.pt hash %s... → %s...",
        old_hash[:12], new_hash[:12],
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # NOTE: We intentionally do NOT update the participation log.
    # The final round's global_model_hash_post still references
    # the original model, creating a detectable inconsistency
    # in the hash chain.  This is itself a finding — the hash
    # chain creates cascading integrity constraints.

    logger.info("  Bundle generated at %s", dest)
    return dest


# ═══════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════


def _rebuild_manifest_from_existing(bundle_dir: Path) -> None:
    """Rebuild manifest.json using metadata from the existing manifest.

    Reads the current manifest to preserve ``run_id``, ``run_seed``,
    timestamps, etc., then recomputes all file hashes from the
    actual files on disk.
    """
    manifest_path = bundle_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        old_manifest = json.load(f)

    build_manifest(
        run_dir=bundle_dir,
        run_id=old_manifest["run_id"],
        run_seed=old_manifest["run_seed"],
        total_rounds=old_manifest["total_rounds_completed"],
        dataset=old_manifest.get("dataset", "cifar10"),
        architecture=old_manifest.get("architecture", "resnet18"),
        start_time=old_manifest.get("start_time"),
        end_time=old_manifest.get("end_time"),
    )
