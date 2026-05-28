"""
verification/verdict.py — Verdict Composition & Fidelity Report
=================================================================

Implements the dual output specified in Section 4.3:

  1. Verification Verdict — binary PASS/FAIL by strict conjunction
     of all active checks at the chosen assurance level.
  2. Unlearning Fidelity Report — per-check deviation magnitudes as
     multiples of the calibrated threshold. Always produced regardless
     of verdict.

The verdict serves the auditor (regulatory compliance).
The fidelity report serves the researcher (comparative evaluation).

Specification references: Sections 4.3, 4.7.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from verification.checks import ASSURANCE_CHECKS, CheckResult


@dataclass
class VerificationVerdict:
    """Binary PASS/FAIL verdict with structured fidelity report.

    Attributes:
        passed: True if ALL active checks passed (conjunction rule).
        assurance_level: The assurance profile used ("basic"/"strong"/"high").
        check_results: Per-check outcomes keyed by check name.
        active_checks: List of check names that were active.
        checks_passed: Count of checks that passed.
        checks_failed: Count of checks that failed.
    """
    passed: bool
    assurance_level: str
    check_results: Dict[str, CheckResult]
    active_checks: List[str]
    checks_passed: int
    checks_failed: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary.

        Returns:
            Nested dict with verdict, per-check results, and summary.
        """
        checks_dict = {}
        for name, result in self.check_results.items():
            checks_dict[name] = {
                "passed": result.passed,
                "measured_value": result.measured_value,
                "threshold": result.threshold,
                "deviation_ratio": result.deviation_ratio,
                "individual_distances": result.individual_distances,
                "gold_reference_medians": result.gold_reference_medians,
                "metadata": result.metadata,
            }

        return {
            "verdict": "PASS" if self.passed else "FAIL",
            "assurance_level": self.assurance_level,
            "checks": checks_dict,
            "summary": {
                "checks_active": len(self.active_checks),
                "checks_passed": self.checks_passed,
                "checks_failed": self.checks_failed,
                "active_check_names": self.active_checks,
                "failed_check_names": [
                    name for name, result in self.check_results.items()
                    if not result.passed
                ],
            },
        }

    def save(self, path: Path) -> None:
        """Write the full verdict + fidelity report to a JSON file.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        verdict_str = "PASS ✓" if self.passed else "FAIL ✗"
        print(f"\n{'=' * 60}")
        print(f"VERIFICATION VERDICT: {verdict_str}")
        print(f"Assurance level: {self.assurance_level}")
        print(f"{'=' * 60}")

        for name in self.active_checks:
            result = self.check_results[name]
            status = "PASS" if result.passed else "FAIL"
            if result.threshold > 0:
                print(
                    f"  {name:30s} {status:4s}  "
                    f"({result.deviation_ratio:.2f}× threshold)"
                )
            else:
                print(f"  {name:30s} {status:4s}")

        print(f"\n  {self.checks_passed}/{len(self.active_checks)} checks passed")
        print(f"{'=' * 60}")


def compose_verdict(
    check_results: Dict[str, CheckResult],
    assurance_level: str,
) -> VerificationVerdict:
    """Compose the verification verdict from individual check results.

    Applies the conjunction rule: all active checks must pass for the
    overall verdict to be PASS.

    Args:
        check_results: Dict mapping check name to its CheckResult.
            Must include results for all checks active at the given
            assurance level.
        assurance_level: The assurance profile ("basic"/"strong"/"high").

    Returns:
        VerificationVerdict with the binary decision and fidelity report.

    Raises:
        ValueError: If the assurance level is not recognised.
        KeyError: If a required check result is missing.
    """
    level = assurance_level.lower()
    if level not in ASSURANCE_CHECKS:
        raise ValueError(
            f"Unknown assurance level '{assurance_level}'. "
            f"Must be one of: {list(ASSURANCE_CHECKS.keys())}"
        )

    active_checks = ASSURANCE_CHECKS[level]

    # Verify all required results are present.
    missing = [c for c in active_checks if c not in check_results]
    if missing:
        raise KeyError(
            f"Missing check results for assurance level '{level}': {missing}"
        )

    # Filter to active checks only.
    active_results = {
        name: check_results[name] for name in active_checks
    }

    # Conjunction rule: all must pass.
    all_passed = all(r.passed for r in active_results.values())
    passed_count = sum(1 for r in active_results.values() if r.passed)
    failed_count = len(active_results) - passed_count

    return VerificationVerdict(
        passed=all_passed,
        assurance_level=level,
        check_results=active_results,
        active_checks=active_checks,
        checks_passed=passed_count,
        checks_failed=failed_count,
    )
