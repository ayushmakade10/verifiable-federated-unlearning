"""
scripts/diagnose_fedosd_selection_seeds.py
===========================================

Diagnostic for the Check 5 ``selection_seeds`` sub-check failure on
FedOSD bundles.

Purpose: confirm *which* rounds break seed-replay and *why*, by
replaying the verifier's exact selection-seed logic round by round
against a real FedOSD participation log. The hypothesis is:

  - UNLEARNING rounds break replay, because FedOSD forces the target
    client into selection and samples 19 (not 20) from the retain
    pool — neither of which the standard replay can reproduce.
  - RECOVERY rounds replay CLEANLY, because they sample 20 from the
    retain pool exactly as the verifier expects.

Usage::

    python scripts/diagnose_fedosd_selection_seeds.py \\
        --bundle outputs/phase4b/fedosd_extended_unlearning \\
        --num-unlearning 20

The --num-unlearning argument tells the diagnostic where the
unlearning/recovery boundary is, so it can label each round.

This script is READ-ONLY. It does not modify any bundle or output.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position
from evidence.participation_log import ParticipationLog
from utils.seeding import derive_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose Check 5 selection-seed replay on a FedOSD bundle.",
    )
    parser.add_argument(
        "--bundle", type=str, required=True,
        help="Path to the FedOSD bundle directory.",
    )
    parser.add_argument(
        "--num-unlearning", type=int, required=True,
        help="Number of unlearning rounds (to label the round boundary).",
    )
    parser.add_argument(
        "--target-client", type=int, default=0,
        help="Target client forced into unlearning rounds (default 0).",
    )
    return parser.parse_args()


def replay_round(
    logged_seed: int,
    available_clients: list[int],
    num_clients: int,
    participation_rate: float,
) -> list[int]:
    """Replay the verifier's EXACT selection logic for one round.

    Mirrors ParticipationLog.verify_selection_seeds (lines 300-302):
        num_selected = max(1, round(num_clients * participation_rate))
        rng = random.Random(logged_seed)
        expected = sorted(rng.sample(available_clients, num_selected))
    """
    num_selected = max(1, round(num_clients * participation_rate))
    rng = random.Random(logged_seed)
    return sorted(rng.sample(available_clients, num_selected))


def main() -> None:
    """Run the round-by-round seed-replay diagnostic."""
    args = parse_args()
    bundle = Path(args.bundle)
    log_path = bundle / "participation_log.json"

    if not log_path.exists():
        print(f"ERROR: participation log not found at {log_path}")
        sys.exit(1)

    log = ParticipationLog.load(str(log_path))

    print("=" * 72)
    print("FedOSD Selection-Seed Replay Diagnostic")
    print("=" * 72)
    print(f"Bundle:              {bundle}")
    print(f"run_seed:            {log.run_seed}")
    print(f"num_clients (log):   {log.num_clients}")
    print(f"participation_rate:  {log.participation_rate}")
    print(f"available_clients:   {len(log.available_clients)} clients "
          f"(target {args.target_client} "
          f"{'PRESENT' if args.target_client in log.available_clients else 'ABSENT'})")
    print(f"total rounds:        {len(log)}")
    print(f"unlearning rounds:   0..{args.num_unlearning - 1}")
    print(f"recovery rounds:     {args.num_unlearning}..{len(log) - 1}")
    print(f"replay draws:        "
          f"{max(1, round(log.num_clients * log.participation_rate))} "
          f"clients from the {len(log.available_clients)}-client pool")
    print("=" * 72)

    num_unlearning = args.num_unlearning
    target = args.target_client

    unlearn_mismatch = 0
    unlearn_total = 0
    recovery_mismatch = 0
    recovery_total = 0
    seed_derivation_ok = True

    print(f"\n{'Round':>5} {'Stage':>10} {'Seed OK':>8} "
          f"{'Logged=Replay':>14} {'Target in logged':>17} {'Count':>6}")
    print("-" * 72)

    for entry in log.rounds:
        rid = entry["round_id"]
        logged_seed = entry["selection_seed"]
        logged_selection = sorted(entry["selected_clients"])

        stage = "unlearn" if rid < num_unlearning else "recovery"

        # 1. Verify seed derivation matches.
        expected_seed = derive_seed(
            log.run_seed, f"client_selection_round_{rid}",
        )
        seed_ok = (logged_seed == expected_seed)
        if not seed_ok:
            seed_derivation_ok = False

        # 2. Replay the selection exactly as the verifier does.
        replayed = replay_round(
            logged_seed,
            log.available_clients,
            log.num_clients,
            log.participation_rate,
        )
        matches = (logged_selection == replayed)

        target_in_logged = target in logged_selection
        count = len(logged_selection)

        if stage == "unlearn":
            unlearn_total += 1
            if not matches:
                unlearn_mismatch += 1
        else:
            recovery_total += 1
            if not matches:
                recovery_mismatch += 1

        # Print first few and any mismatches of each stage.
        show = (
            rid < 3
            or rid == num_unlearning
            or rid == num_unlearning + 1
            or (stage == "unlearn" and matches)
            or (stage == "recovery" and not matches)
        )
        if show:
            print(f"{rid:>5} {stage:>10} {'yes' if seed_ok else 'NO':>8} "
                  f"{'MATCH' if matches else 'MISMATCH':>14} "
                  f"{'yes' if target_in_logged else 'no':>17} {count:>6}")

    print("-" * 72)
    print("\nSUMMARY")
    print("=" * 72)
    print(f"Seed derivation (all rounds):  "
          f"{'OK — all seeds derived correctly' if seed_derivation_ok else 'BROKEN'}")
    print(f"Unlearning rounds:  {unlearn_mismatch}/{unlearn_total} MISMATCH "
          f"(replay cannot reproduce forced target + 19-from-retain)")
    print(f"Recovery rounds:    {recovery_mismatch}/{recovery_total} MISMATCH "
          f"(should be 0 — these sample 20 from retain, as verifier expects)")
    print("=" * 72)

    # Verdict on the hypothesis.
    print("\nHYPOTHESIS CHECK")
    print("-" * 72)
    hyp_unlearn = (unlearn_mismatch == unlearn_total and unlearn_total > 0)
    hyp_recovery = (recovery_mismatch == 0)

    print(f"  [{'PASS' if hyp_unlearn else 'FAIL'}] "
          f"ALL unlearning rounds break seed-replay")
    print(f"  [{'PASS' if hyp_recovery else 'FAIL'}] "
          f"NO recovery rounds break seed-replay")

    if hyp_unlearn and hyp_recovery:
        print("\n  CONFIRMED: The selection_seeds failure is driven entirely by")
        print("  the unlearning rounds' forced target-client inclusion. Recovery")
        print("  rounds replay cleanly. This is the inferred mechanism, proven.")
    else:
        print("\n  The mechanism differs from the inferred story — see per-round")
        print("  detail above. The verdict (FAIL) is unaffected, but the")
        print("  explanation needs revising.")
    print("=" * 72)


if __name__ == "__main__":
    main()
