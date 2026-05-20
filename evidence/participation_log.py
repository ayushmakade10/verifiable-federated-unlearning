"""
evidence/participation_log.py — Participation Log Access Layer
================================================================

Flat JSON file on disk + in-memory class with indexed query methods.
The class serves two roles:

  1. **Builder** (during training): accumulate round entries via
     add_round(), then save() to disk.
  2. **Query** (during verification): load from disk, then use
     indexed lookups for O(1) access by round or client.

Specification reference: Sections 6.3 and 7.2 of the master document.

JSON structure on disk:
    {
      "run_id": "run_001",
      "run_seed": 42,
      "num_clients": 50,
      "participation_rate": 0.4,
      "rounds": [ { round entry }, ... ]
    }

Each round entry:
    {
      "round_id": 0,
      "selection_seed": 1938475,
      "selected_clients": [3, 7, 14, 22, ...],
      "num_samples_per_client": {"3": 847, "7": 1023, ...},
      "global_model_hash_pre": "a3f8c9...",
      "global_model_hash_post": "b7d2e1...",
      "test_accuracy": 0.423,
      "test_loss": 1.847
    }

Usage:
    # Building during training:
    log = ParticipationLog(run_id="run_001", run_seed=42,
                           num_clients=50, participation_rate=0.4)
    log.add_round(round_id=0, selection_seed=..., ...)
    log.save("outputs/run_001/participation_log.json")

    # Querying during verification:
    log = ParticipationLog.load("outputs/run_001/participation_log.json")
    rounds_for_client_7 = log.get_client_rounds(7)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.seeding import derive_seed


class ParticipationLog:
    """In-memory representation of the participation log with indexed queries.

    Internal indices are built lazily on first query or explicitly via
    _build_indices(). This keeps construction cheap during training
    (when we only append) while still providing O(1) lookups during
    verification.
    """

    def __init__(
        self,
        run_id: str,
        run_seed: int,
        num_clients: int,
        participation_rate: float,
        rounds: Optional[List[Dict[str, Any]]] = None,
        available_clients: Optional[List[int]] = None,
    ) -> None:
        self.run_id = run_id
        self.run_seed = run_seed
        self.num_clients = num_clients
        self.participation_rate = participation_rate
        self._rounds: List[Dict[str, Any]] = list(rounds) if rounds else []

        # The actual client pool used for selection. For original runs
        # this is [0..num_clients-1]. For gold retraining runs with a
        # removed client, this has a gap (e.g. [0..6, 8..49]).
        # Critical for verify_selection_seeds() correctness.
        if available_clients is not None:
            self.available_clients = sorted(available_clients)
        else:
            # Fallback: assume consecutive IDs (original run).
            self.available_clients = list(range(num_clients))

        # Indices — built lazily.
        self._round_index: Optional[Dict[int, Dict[str, Any]]] = None
        self._client_index: Optional[Dict[int, List[int]]] = None

    # ── Builder methods (used during training) ───────────────────

    def add_round(
        self,
        round_id: int,
        selection_seed: int,
        selected_clients: List[int],
        num_samples_per_client: Dict[int, int],
        global_model_hash_pre: str,
        global_model_hash_post: str,
        test_accuracy: float,
        test_loss: float,
    ) -> None:
        """Append a round entry to the log.

        Invalidates cached indices so the next query rebuilds them.

        Args:
            round_id:                The round number (0-indexed).
            selection_seed:          The seed used for client selection this round.
            selected_clients:        List of selected client IDs.
            num_samples_per_client:  Map from client_id → number of training samples.
            global_model_hash_pre:   SHA-256 of global model before aggregation.
            global_model_hash_post:  SHA-256 of global model after aggregation.
            test_accuracy:           Test set accuracy after this round.
            test_loss:               Test set loss after this round.
        """
        entry = {
            "round_id": round_id,
            "selection_seed": selection_seed,
            "selected_clients": sorted(selected_clients),
            "num_samples_per_client": {
                str(k): v for k, v in num_samples_per_client.items()
            },
            "global_model_hash_pre": global_model_hash_pre,
            "global_model_hash_post": global_model_hash_post,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
        }
        self._rounds.append(entry)

        # Invalidate indices.
        self._round_index = None
        self._client_index = None

    # ── Serialisation ────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return the full log as a JSON-serialisable dictionary."""
        return {
            "run_id": self.run_id,
            "run_seed": self.run_seed,
            "num_clients": self.num_clients,
            "available_clients": self.available_clients,
            "participation_rate": self.participation_rate,
            "rounds": self._rounds,
        }

    def save(self, path: str | Path) -> None:
        """Write the log to a JSON file on disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ParticipationLog":
        """Load a participation log from a JSON file.

        Returns:
            A ParticipationLog instance with indices ready for queries.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Participation log not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            run_id=data["run_id"],
            run_seed=data["run_seed"],
            num_clients=data["num_clients"],
            participation_rate=data["participation_rate"],
            rounds=data["rounds"],
            available_clients=data.get("available_clients"),  # None → fallback
        )

    # ── Index building ───────────────────────────────────────────

    def _build_indices(self) -> None:
        """Build internal lookup structures from the round list."""
        self._round_index = {}
        self._client_index = {}

        for entry in self._rounds:
            rid = entry["round_id"]
            self._round_index[rid] = entry
            for cid in entry["selected_clients"]:
                self._client_index.setdefault(cid, []).append(rid)

    def _ensure_indices(self) -> None:
        """Build indices if they haven't been built yet."""
        if self._round_index is None or self._client_index is None:
            self._build_indices()

    # ── Query methods (used during verification) ─────────────────

    @property
    def rounds(self) -> List[Dict[str, Any]]:
        """Sequential access to all round entries (read-only copy)."""
        return list(self._rounds)

    def get_round(self, round_id: int) -> Dict[str, Any]:
        """Get the full entry for a specific round.

        Args:
            round_id: The round number to look up.

        Returns:
            The round entry dictionary.

        Raises:
            KeyError: If the round_id is not in the log.
        """
        self._ensure_indices()
        assert self._round_index is not None  # for type checker
        if round_id not in self._round_index:
            raise KeyError(f"Round {round_id} not found in participation log")
        return self._round_index[round_id]

    def get_client_rounds(self, client_id: int) -> List[int]:
        """Get all round IDs where the given client participated.

        Args:
            client_id: The client to look up.

        Returns:
            Sorted list of round IDs. Empty list if client never participated.
        """
        self._ensure_indices()
        assert self._client_index is not None  # for type checker
        return sorted(self._client_index.get(client_id, []))

    def get_first_participation(self, client_id: int) -> Optional[int]:
        """Get the earliest round where the client participated.

        Args:
            client_id: The client to look up.

        Returns:
            The round ID, or None if the client never participated.
        """
        rounds = self.get_client_rounds(client_id)
        return rounds[0] if rounds else None

    # ── Verification methods ─────────────────────────────────────

    def verify_hash_chain(self) -> bool:
        """Verify that the model hash chain is consistent.

        For consecutive rounds r and r+1, checks that:
            global_model_hash_post[r] == global_model_hash_pre[r+1]

        This ensures the global model was not tampered with between
        rounds during training.

        Returns:
            True if the chain is consistent, False if any break is found.
        """
        sorted_rounds = sorted(self._rounds, key=lambda e: e["round_id"])
        for i in range(len(sorted_rounds) - 1):
            current = sorted_rounds[i]
            next_round = sorted_rounds[i + 1]

            # Only check consecutive rounds (there may be gaps in resumed runs).
            if next_round["round_id"] == current["round_id"] + 1:
                if current["global_model_hash_post"] != next_round["global_model_hash_pre"]:
                    return False
        return True

    def verify_selection_seeds(self, run_seed: int) -> bool:
        """Recompute client selections from seeds and verify they match the log.

        For each round, derives the selection seed from the run_seed and
        round number, then resamples clients. Checks that the logged
        selected_clients match the recomputed selection.

        Args:
            run_seed: The run-level seed that was used for training.

        Returns:
            True if all round selections match, False otherwise.
        """
        for entry in self._rounds:
            round_id = entry["round_id"]
            logged_seed = entry["selection_seed"]

            # Verify the seed itself was derived correctly.
            expected_seed = derive_seed(run_seed, f"client_selection_round_{round_id}")
            if logged_seed != expected_seed:
                return False

            # Verify the client selection matches.
            num_selected = max(1, round(self.num_clients * self.participation_rate))
            rng = random.Random(logged_seed)
            expected_selection = sorted(rng.sample(self.available_clients, num_selected))

            if sorted(entry["selected_clients"]) != expected_selection:
                return False

        return True

    def __len__(self) -> int:
        return len(self._rounds)

    def __repr__(self) -> str:
        return (
            f"ParticipationLog(run_id={self.run_id!r}, "
            f"rounds={len(self._rounds)}, "
            f"clients={self.num_clients})"
        )


# ── Verification ─────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("participation_log.py: running verification\n")

    # 1. Build a log (original run — consecutive client IDs, fallback path).
    log = ParticipationLog(
        run_id="test_run",
        run_seed=42,
        num_clients=10,
        participation_rate=0.4,
    )
    assert log.available_clients == list(range(10)), "Fallback should be [0..9]"
    log.add_round(
        round_id=0,
        selection_seed=derive_seed(42, "client_selection_round_0"),
        selected_clients=[1, 3, 5, 7],
        num_samples_per_client={1: 100, 3: 200, 5: 150, 7: 180},
        global_model_hash_pre="aaa",
        global_model_hash_post="bbb",
        test_accuracy=0.15,
        test_loss=2.3,
    )
    log.add_round(
        round_id=1,
        selection_seed=derive_seed(42, "client_selection_round_1"),
        selected_clients=[0, 2, 5, 9],
        num_samples_per_client={0: 120, 2: 180, 5: 150, 9: 90},
        global_model_hash_pre="bbb",
        global_model_hash_post="ccc",
        test_accuracy=0.20,
        test_loss=2.1,
    )
    assert len(log) == 2
    print("[1/8] Built log with 2 rounds")

    # 2. Query by round.
    r0 = log.get_round(0)
    assert r0["test_accuracy"] == 0.15
    print("[2/8] get_round() works")

    # 3. Query by client.
    assert log.get_client_rounds(5) == [0, 1]
    assert log.get_client_rounds(7) == [0]
    assert log.get_client_rounds(8) == []
    print("[3/8] get_client_rounds() works")

    # 4. First participation.
    assert log.get_first_participation(5) == 0
    assert log.get_first_participation(9) == 1
    assert log.get_first_participation(8) is None
    print("[4/8] get_first_participation() works")

    # 5. Hash chain verification.
    assert log.verify_hash_chain() is True
    print("[5/8] Hash chain verified (consistent)")

    # 6. Save/load round-trip.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    log.save(tmp)
    loaded = ParticipationLog.load(tmp)
    assert len(loaded) == 2
    assert loaded.get_round(0)["test_accuracy"] == 0.15
    assert loaded.get_client_rounds(5) == [0, 1]
    assert loaded.available_clients == list(range(10))
    print(f"[6/8] Save/load round-trip via {tmp}")

    # 7. available_clients persists through save/load.
    with open(tmp, encoding="utf-8") as f:
        raw = json.load(f)
    assert "available_clients" in raw, "available_clients must be in JSON"
    assert raw["available_clients"] == list(range(10))
    print("[7/8] available_clients serialised and loaded correctly")

    # 8. Gold retraining case — gap in client pool.
    #    Simulate removing client 5 from a pool of 10.
    gold_clients = [0, 1, 2, 3, 4, 6, 7, 8, 9]  # client 5 removed
    gold_log = ParticipationLog(
        run_id="gold_test",
        run_seed=99,
        num_clients=9,  # 9 remaining clients
        participation_rate=0.4,
        available_clients=gold_clients,
    )
    assert gold_log.available_clients == gold_clients

    # Build a round using the gapped client pool.
    sel_seed = derive_seed(99, "client_selection_round_0")
    rng = random.Random(sel_seed)
    num_sel = max(1, round(9 * 0.4))
    selected = sorted(rng.sample(gold_clients, num_sel))

    gold_log.add_round(
        round_id=0,
        selection_seed=sel_seed,
        selected_clients=selected,
        num_samples_per_client={c: 100 for c in selected},
        global_model_hash_pre="xxx",
        global_model_hash_post="yyy",
        test_accuracy=0.80,
        test_loss=0.6,
    )
    # verify_selection_seeds must pass with the gapped pool.
    assert gold_log.verify_selection_seeds(99) is True, \
        "verify_selection_seeds must work with gapped client pool"
    # And must fail with the wrong seed.
    assert gold_log.verify_selection_seeds(100) is False
    print("[8/8] Gold retraining (gapped client pool) verified correctly")

    print("\nparticipation_log.py: all checks passed ✓")
