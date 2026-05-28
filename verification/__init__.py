"""
verification/ — Verification Module Public API
================================================

Phase 3 of the dissertation: implements the five verification checks,
probe set construction, threshold calibration, and verdict assembly.

Primary entry points:

  - run_verification(): Full pipeline from evidence bundle to verdict.
  - compose_verdict(): Combine check results into binary decision.
  - CalibrationBundle: Load/save calibrated thresholds.
  - CheckResult: Standard result type for all checks.
  - ASSURANCE_CHECKS: Which checks are active per assurance level.

Specification references: Sections 4.1–4.8, 9.2.
"""

from verification.calibration import CalibrationBundle, CheckCalibration
from verification.checks import ASSURANCE_CHECKS, CheckResult
from verification.comparison import ComparisonResult, compare_against_gold
from verification.runner import run_verification
from verification.verdict import VerificationVerdict, compose_verdict

__all__ = [
    "ASSURANCE_CHECKS",
    "CalibrationBundle",
    "CheckCalibration",
    "CheckResult",
    "ComparisonResult",
    "VerificationVerdict",
    "compare_against_gold",
    "compose_verdict",
    "run_verification",
]
