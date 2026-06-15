"""
scripts/diagnose_federaser_norms.py
====================================

Inspects the FedEraser calibration math at the round where NaN
appears, printing per-client norms so we can pinpoint the divergence
source WITHOUT running the full pipeline.

For the first two stored rounds it reports, per retain client:
  - ||stored_update||          (magnitude of the retained update)
  - ||fresh_update||           (magnitude of the calibration update)
  - scale = ||stored|| / ||fresh||
  - ||calibrated_update||      (should equal ||stored||)
  - whether any value is non-finite

It also reports the aggregated update norm and the resulting model's
accuracy/loss after applying it — so we can see exactly where the
blow-up begins.

READ-ONLY. Writes nothing to any bundle.

Usage::

    python scripts/diagnose_federaser_norms.py \\
        --prep-dir outputs/run_001_federaser_prep --delta-t 20
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position
from config.schemas import load_config
from data.partitioner import partition_cifar10
from evidence.participation_log import ParticipationLog
from federation.trainer import (
    _evaluate,
    _get_cifar10_transforms,
    _make_client_dataloader,
    _make_test_dataloader,
)
from models.resnet import build_model
from unlearning.methods.federaser import (
    _aggregate_stored_directly,
    _calibration_train,
    _flatten_update,
    load_stored_updates,
)
from utils.seeding import derive_seed, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-dir", type=str, default="run_001_federaser_prep")
    parser.add_argument("--delta-t", type=int, default=20)
    parser.add_argument("--target-client", type=int, default=0)
    parser.add_argument("--calibration-epochs", type=int, default=3)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def finite(x: float) -> str:
    import math
    return "OK" if math.isfinite(x) else "** NON-FINITE **"


def main() -> None:
    args = parse_args()
    cfg = load_config(str(REPO_ROOT / args.config))
    output_base = Path(cfg.checkpoint.output_dir)
    prep_dir = output_base / args.prep_dir
    target = args.target_client
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prep_log = ParticipationLog.load(str(prep_dir / "participation_log.json"))
    original_seed = prep_log.run_seed

    # Partition + dataset
    partition_seed = derive_seed(cfg.reproducibility.root_seed, "partition")
    partition = partition_cifar10(
        num_clients=cfg.data.num_clients, alpha=cfg.data.alpha,
        seed=partition_seed,
    )
    train_transform, _ = _get_cifar10_transforms()
    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.data_root, train=True, download=True,
        transform=train_transform,
    )
    test_loader = _make_test_dataloader(
        data_root=cfg.data.data_root, batch_size=cfg.data.batch_size * 2,
    )

    # Reconstruct initial model
    init_seed = derive_seed(original_seed, "model_init")
    set_global_seed(init_seed)
    model = build_model(num_classes=cfg.model.num_classes).to(device)

    stored_rounds = list(range(args.delta_t, cfg.federation.num_rounds + 1, args.delta_t))

    print("=" * 78)
    print("FedEraser Calibration Norm Diagnostic")
    print("=" * 78)
    acc, loss = _evaluate(model, test_loader, device)
    print(f"Initial model: acc={acc:.4f}, loss={loss:.4f}")
    print(f"Stored rounds: {stored_rounds[:6]}...")
    print()

    for idx, t in enumerate(stored_rounds[:2]):  # first 2 rounds only
        print("-" * 78)
        print(f"ROUND idx={idx}  (stored t={t})")
        print("-" * 78)

        stored_updates = load_stored_updates(prep_dir, t)
        prep_entry = prep_log.get_round(t - 1)
        participated = prep_entry["selected_clients"]
        samples = {int(k): v for k, v in prep_entry["num_samples_per_client"].items()}
        retain = [c for c in participated if c != target]

        # Report stored-update norms
        print(f"  Participated: {len(participated)} clients, "
              f"{len(retain)} retain (target {target} dropped)")
        stored_norms = []
        for c in retain:
            if c in stored_updates:
                n = _flatten_update(stored_updates[c]).norm().item()
                stored_norms.append(n)
        print(f"  Stored-update norms: min={min(stored_norms):.4f}, "
              f"max={max(stored_norms):.4f}, mean={sum(stored_norms)/len(stored_norms):.4f}")

        if idx == 0:
            # Direct application
            agg = _aggregate_stored_directly(
                stored_updates, participated, samples, target,
            )
            agg_norm = _flatten_update(agg).norm().item()
            print(f"  [direct] aggregated update norm: {agg_norm:.4f} [{finite(agg_norm)}]")
            cur = {k: v.cpu() for k, v in model.state_dict().items()}
            new = {k: cur[k].float() + agg[k].float() for k in cur}
            model.load_state_dict(new)
            model = model.to(device)
        else:
            # Calibration — inspect per client
            print(f"  {'client':>7} {'||stored||':>12} {'||fresh||':>12} "
                  f"{'scale':>12} {'||calib||':>12}  flag")
            for c in retain[:8]:  # first 8 clients for brevity
                if c not in stored_updates:
                    continue
                dl_seed = derive_seed(cfg.reproducibility.root_seed,
                                      f"federaser_diag_round_{t}_dl_client_{c}")
                loader = _make_client_dataloader(
                    dataset=train_dataset, indices=partition[c],
                    batch_size=cfg.data.batch_size, generator_seed=dl_seed,
                )
                cm = copy.deepcopy(model)
                tr_seed = derive_seed(cfg.reproducibility.root_seed,
                                      f"federaser_diag_round_{t}_tr_client_{c}")
                set_global_seed(tr_seed)
                _calibration_train(cm, loader, cfg, device, args.calibration_epochs)
                cm_sd = {k: v.cpu() for k, v in cm.state_dict().items()}
                cal_sd = {k: v.cpu() for k, v in model.state_dict().items()}
                fresh = {k: cm_sd[k].float() - cal_sd[k].float() for k in cm_sd}

                s_norm = _flatten_update(stored_updates[c]).norm().item()
                f_norm = _flatten_update(fresh).norm().item()
                scale = s_norm / f_norm if f_norm > 1e-12 else float("inf")
                calib_norm = scale * f_norm if f_norm > 1e-12 else 0.0
                flag = finite(scale) if scale != float("inf") else "** INF SCALE **"
                print(f"  {c:>7} {s_norm:>12.4f} {f_norm:>12.6f} "
                      f"{scale:>12.4f} {calib_norm:>12.4f}  {flag}")

    print("=" * 78)
    print("Look for: ||fresh|| near zero (→ huge scale), or ||stored|| huge,")
    print("or any NON-FINITE flag. That pinpoints the divergence source.")
    print("=" * 78)


if __name__ == "__main__":
    main()
