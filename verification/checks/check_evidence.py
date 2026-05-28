"""
verification/checks/check_evidence.py — Check 5: Evidence Consistency
========================================================================

Verifies the integrity and consistency of the evidence bundle at
three strictness tiers, parameterised by assurance level:

  Basic:  Hash chain integrity + selection seed verification + round count
  Strong: + metadata cross-check against config + checkpoint existence
  High:   + full manifest SHA-256 recomputation of every file

A single implementation handles all three tiers. The assurance level
determines which sub-checks are active.

Detects: tampered evidence, inconsistent logs, forged participation
records, modified checkpoint files.

Specification reference: Sections 4.8 (Check 5), 6.1–6.4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from evidence.hashing import hash_file
from evidence.participation_log import ParticipationLog
from verification.checks import CheckResult


def run_check(
    bundle_path: Path,
    assurance_level: str,
    expected_num_rounds: int,
    expected_num_clients: int,
    expected_participation_rate: float,
    expected_checkpoint_interval: int,
) -> CheckResult:
    """Run Check 5: Participation & Evidence Consistency.

    Args:
        bundle_path: Path to the evidence bundle directory
            (e.g. outputs/run_001/).
        assurance_level: "basic", "strong", or "high".
        expected_num_rounds: Expected total rounds from config.
        expected_num_clients: Expected number of clients from config.
        expected_participation_rate: Expected participation rate from config.
        expected_checkpoint_interval: Expected checkpoint interval from config.

    Returns:
        CheckResult with pass/fail and per-tier detail in metadata.
    """
    failures: List[str] = []
    tier_results: Dict[str, Dict[str, Any]] = {}

    # ── Load participation log ───────────────────────────────────
    log_path = bundle_path / "participation_log.json"
    if not log_path.exists():
        return _fail("Participation log not found", bundle_path)

    log = ParticipationLog.load(str(log_path))

    # ── Load manifest ────────────────────────────────────────────
    manifest_path = bundle_path / "manifest.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    # ── Basic tier ───────────────────────────────────────────────
    basic_results = _check_basic(log, expected_num_rounds)
    tier_results["basic"] = basic_results
    if not basic_results["passed"]:
        failures.extend(basic_results["failures"])

    # ── Strong tier (adds metadata cross-check + checkpoint existence)
    if assurance_level in ("strong", "high"):
        strong_results = _check_strong(
            log, bundle_path, manifest,
            expected_num_clients,
            expected_participation_rate,
            expected_num_rounds,
            expected_checkpoint_interval,
        )
        tier_results["strong"] = strong_results
        if not strong_results["passed"]:
            failures.extend(strong_results["failures"])

    # ── High tier (adds full manifest hash verification) ─────────
    if assurance_level == "high":
        high_results = _check_high(bundle_path, manifest)
        tier_results["high"] = high_results
        if not high_results["passed"]:
            failures.extend(high_results["failures"])

    passed = len(failures) == 0

    return CheckResult(
        check_name="evidence_consistency",
        passed=passed,
        measured_value=0.0 if passed else 1.0,
        threshold=0.0,
        deviation_ratio=0.0,
        metadata={
            "assurance_level": assurance_level,
            "tier_results": tier_results,
            "failures": failures,
            "note": (
                "Check 5 is binary (pass/fail per tier), not a "
                "continuous distance metric. measured_value=0 means "
                "all tiers passed; measured_value=1 means at least "
                "one tier failed."
            ),
        },
    )


# ── Basic Tier ───────────────────────────────────────────────────


def _check_basic(
    log: ParticipationLog,
    expected_num_rounds: int,
) -> Dict[str, Any]:
    """Basic tier: hash chain + seed verification + round count.

    Args:
        log: Loaded participation log.
        expected_num_rounds: Expected total rounds.

    Returns:
        Dict with 'passed', 'failures', and sub-check details.
    """
    failures: List[str] = []
    sub_checks: Dict[str, bool] = {}

    # 1. Hash chain integrity.
    hash_chain_ok = log.verify_hash_chain()
    sub_checks["hash_chain"] = hash_chain_ok
    if not hash_chain_ok:
        failures.append("Hash chain broken: post-hash of round r != pre-hash of round r+1")

    # 2. Selection seed verification.
    seed_ok = log.verify_selection_seeds(log.run_seed)
    sub_checks["selection_seeds"] = seed_ok
    if not seed_ok:
        failures.append(
            "Selection seed verification failed: logged selections "
            "don't match recomputed"
        )

    # 3. Round count.
    actual_rounds = len(log)
    round_count_ok = actual_rounds == expected_num_rounds
    sub_checks["round_count"] = round_count_ok
    if not round_count_ok:
        failures.append(
            f"Round count mismatch: log has {actual_rounds}, "
            f"expected {expected_num_rounds}"
        )

    return {
        "passed": len(failures) == 0,
        "sub_checks": sub_checks,
        "failures": failures,
    }


# ── Strong Tier ──────────────────────────────────────────────────


def _check_strong(
    log: ParticipationLog,
    bundle_path: Path,
    manifest: dict | None,
    expected_num_clients: int,
    expected_participation_rate: float,
    expected_num_rounds: int,
    expected_checkpoint_interval: int,
) -> Dict[str, Any]:
    """Strong tier: metadata cross-check + checkpoint file existence.

    Args:
        log: Loaded participation log.
        bundle_path: Path to the evidence bundle directory.
        manifest: Loaded manifest dict (may be None).
        expected_num_clients: From config.
        expected_participation_rate: From config.
        expected_num_rounds: From config.
        expected_checkpoint_interval: From config.

    Returns:
        Dict with 'passed', 'failures', and sub-check details.
    """
    failures: List[str] = []
    sub_checks: Dict[str, bool] = {}

    # 1. Log metadata matches config.
    clients_ok = log.num_clients == expected_num_clients
    sub_checks["num_clients"] = clients_ok
    if not clients_ok:
        failures.append(
            f"Client count mismatch: log={log.num_clients}, "
            f"config={expected_num_clients}"
        )

    rate_ok = abs(log.participation_rate - expected_participation_rate) < 1e-6
    sub_checks["participation_rate"] = rate_ok
    if not rate_ok:
        failures.append(
            f"Participation rate mismatch: log={log.participation_rate}, "
            f"config={expected_participation_rate}"
        )

    # 2. Manifest presence.
    manifest_ok = manifest is not None
    sub_checks["manifest_present"] = manifest_ok
    if not manifest_ok:
        failures.append("Manifest file (manifest.json) not found in bundle")

    # 3. Checkpoint files exist at expected intervals.
    ckpt_dir = bundle_path / "checkpoints"
    expected_rounds = list(
        range(
            expected_checkpoint_interval,
            expected_num_rounds + 1,
            expected_checkpoint_interval,
        )
    )
    missing_ckpts = []
    for rnd in expected_rounds:
        ckpt_path = ckpt_dir / f"round_{rnd:03d}.pt"
        if not ckpt_path.exists():
            missing_ckpts.append(rnd)

    ckpts_ok = len(missing_ckpts) == 0
    sub_checks["checkpoints_exist"] = ckpts_ok
    if not ckpts_ok:
        failures.append(
            f"Missing checkpoint files for rounds: {missing_ckpts}"
        )

    # 4. Manifest total_rounds matches config.
    if manifest is not None:
        manifest_rounds = manifest.get("total_rounds_completed", -1)
        rounds_ok = manifest_rounds == expected_num_rounds
        sub_checks["manifest_rounds"] = rounds_ok
        if not rounds_ok:
            failures.append(
                f"Manifest rounds mismatch: manifest={manifest_rounds}, "
                f"config={expected_num_rounds}"
            )

    return {
        "passed": len(failures) == 0,
        "sub_checks": sub_checks,
        "failures": failures,
        "missing_checkpoints": missing_ckpts,
    }


# ── High Tier ────────────────────────────────────────────────────


def _check_high(
    bundle_path: Path,
    manifest: dict | None,
) -> Dict[str, Any]:
    """High tier: full manifest SHA-256 hash recomputation.

    Recomputes the SHA-256 hash of every file listed in the manifest
    and verifies it matches exactly. Any single mismatch is an
    immediate failure.

    Args:
        bundle_path: Path to the evidence bundle directory.
        manifest: Loaded manifest dict.

    Returns:
        Dict with 'passed', 'failures', and per-file verification.
    """
    failures: List[str] = []
    file_checks: Dict[str, Dict[str, Any]] = {}

    if manifest is None:
        return {
            "passed": False,
            "failures": ["Cannot verify hashes: manifest.json not found"],
            "file_checks": {},
        }

    file_hashes = manifest.get("file_hashes", {})
    if not file_hashes:
        return {
            "passed": False,
            "failures": ["Manifest contains no file_hashes entries"],
            "file_checks": {},
        }

    for filename, expected_hash in file_hashes.items():
        filepath = bundle_path / filename
        if not filepath.exists():
            file_checks[filename] = {
                "status": "missing",
                "expected": expected_hash[:16] + "...",
            }
            failures.append(f"File missing: {filename}")
            continue

        actual_hash = hash_file(filepath)
        matches = actual_hash == expected_hash
        file_checks[filename] = {
            "status": "match" if matches else "mismatch",
            "expected": expected_hash[:16] + "...",
            "actual": actual_hash[:16] + "...",
        }
        if not matches:
            failures.append(f"Hash mismatch: {filename}")

    return {
        "passed": len(failures) == 0,
        "file_checks": file_checks,
        "failures": failures,
        "files_verified": len(file_hashes),
        "files_matched": sum(
            1 for v in file_checks.values() if v["status"] == "match"
        ),
    }


# ── Helpers ──────────────────────────────────────────────────────


def _fail(reason: str, bundle_path: Path) -> CheckResult:
    """Create an immediate failure CheckResult."""
    return CheckResult(
        check_name="evidence_consistency",
        passed=False,
        measured_value=1.0,
        threshold=0.0,
        deviation_ratio=0.0,
        metadata={
            "status": "critical_failure",
            "reason": reason,
            "bundle_path": str(bundle_path),
        },
    )
