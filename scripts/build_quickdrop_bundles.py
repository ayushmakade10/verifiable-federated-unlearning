"""
scripts/build_quickdrop_bundles.py — QuickDrop (Method 5) Evidence Bundles
===========================================================================

Assembles evidence bundles for the three QuickDrop provider configurations
(default / extended / finetuned) under ``outputs/phase4b_quickdrop/``, so the
verification engine can evaluate them. Follows the SGA bundle template
(``_assemble_sga_bundle``) exactly:

    outputs/phase4b_quickdrop/<config>/
    ├── final_model.pt          (the provider model state_dict)
    ├── config.yaml             (frozen QuickDrop config; architecture=convnet)
    ├── participation_log.json  (0 FL rounds — like SGA; no honest-retrain log)
    ├── unlearning_request.json (target f_00000 + its class histogram)
    └── manifest.json           (SHA-256 of every file above)

No ``checkpoints/`` directory is created — consistent with Methods 1-4 and the
approved decision that Check 4 reports ``insufficient_evidence`` for QuickDrop
(the unlearn+recover process produces no 200-round honest-retraining trajectory).

The provider models and the partition (for the target's class histogram) are
read from the QuickDrop sibling repo; bundles are written into the dissertation
repo's outputs tree. Additive; touches no existing file.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from evidence.bundle import build_manifest  # noqa: E402
from evidence.participation_log import ParticipationLog  # noqa: E402

# (config_name, provider model filename) for the three configs.
CONFIGS = [
    ("quickdrop_default", "provider_default.pt"),
    ("quickdrop_extended", "provider_extended.pt"),
    ("quickdrop_finetuned", "provider_finetuned.pt"),
]
TARGET_CLIENT = 0
TARGET_USER = "f_00000"
NUM_CLASSES = 10
PARTITION_REL = Path("env") / "dilichlet" / "CIFAR10" / \
    "cifar10-seed42-u50-alpha0.3" / "train" / "train.pt"
PROVIDER_REL = Path("vfu_outputs") / "provider"


def compute_class_histogram(partition_path: Path) -> list[int]:
    """Per-class training-sample counts for the target client (f_00000).

    Computed from the QuickDrop partition — the world the ConvNet gold and
    provider were trained in — so the probe set is faithful to the target
    client's actual class distribution.
    """
    data = torch.load(partition_path, map_location="cpu", weights_only=False)
    if TARGET_USER not in data["user_data"]:
        raise KeyError(f"{TARGET_USER} not in partition {partition_path}")
    y = data["user_data"][TARGET_USER]["y"].long()
    return torch.bincount(y, minlength=NUM_CLASSES).tolist()


def assemble_bundle(
    provider_pt: Path,
    run_dir: Path,
    run_id: str,
    config_dict: dict,
    histogram: list[int],
) -> None:
    """Assemble one QuickDrop evidence bundle (SGA-template structure)."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. final_model.pt — the provider state_dict (verified pure state_dict).
    sd = torch.load(provider_pt, map_location="cpu", weights_only=True)
    torch.save(sd, run_dir / "final_model.pt")

    # 2. Frozen config (architecture=convnet).
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # 3. Participation log — 0 FL rounds (no honest-retraining trajectory),
    #    consistent with the SGA bundle and our Check 4/5 decision.
    log = ParticipationLog(
        run_id=run_id,
        run_seed=0,
        num_clients=config_dict["data"]["num_clients"],
        participation_rate=config_dict["federation"]["participation_rate"],
        rounds=[],
    )
    log.save(run_dir / "participation_log.json")

    # 4. Unlearning request (target + class histogram for the probe set).
    request = {
        "request_id": f"req_{TARGET_CLIENT:03d}",
        "target_client_id": TARGET_CLIENT,
        "source_run_id": "quickdrop_original",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_type": "client_deletion",
        "client_class_histogram": histogram,
    }
    with open(run_dir / "unlearning_request.json", "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)

    # 5. Manifest (SHA-256 of every file above).
    build_manifest(
        run_dir=run_dir,
        run_id=run_id,
        run_seed=0,
        total_rounds=0,
        architecture="convnet",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build QuickDrop (Method 5) evidence bundles."
    )
    ap.add_argument(
        "--quickdrop-root", type=str,
        default=str(REPO_ROOT.parent / "quickdrop-main" / "quickdrop-main"),
        help="Path to the QuickDrop sibling repo (holds vfu_outputs + env).",
    )
    ap.add_argument(
        "--output-base", type=str, default=str(REPO_ROOT / "outputs"),
        help="Output base; bundles go under <base>/phase4b_quickdrop/.",
    )
    ap.add_argument(
        "--config", type=str, default=str(REPO_ROOT / "config" / "quickdrop.yaml"),
        help="QuickDrop config to freeze into each bundle.",
    )
    args = ap.parse_args()

    quickdrop_root = Path(args.quickdrop_root)
    output_base = Path(args.output_base)
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Target client class histogram (from the QuickDrop partition).
    partition_path = quickdrop_root / PARTITION_REL
    if not partition_path.exists():
        raise FileNotFoundError(f"QuickDrop partition not found: {partition_path}")
    histogram = compute_class_histogram(partition_path)
    print(f"[bundles] target {TARGET_USER} class histogram: {histogram} "
          f"(total {sum(histogram)} samples)")

    provider_dir = quickdrop_root / PROVIDER_REL
    bundles_root = output_base / "phase4b_quickdrop"
    for config_name, provider_file in CONFIGS:
        provider_pt = provider_dir / provider_file
        if not provider_pt.exists():
            raise FileNotFoundError(f"Provider model not found: {provider_pt}")
        run_dir = bundles_root / config_name
        assemble_bundle(
            provider_pt=provider_pt,
            run_dir=run_dir,
            run_id=f"phase4b_quickdrop/{config_name}",
            config_dict=config_dict,
            histogram=histogram,
        )
        files = sorted(p.name for p in run_dir.iterdir())
        print(f"[bundles] {config_name}: {files}")

    print(f"[bundles] done — 3 bundles under {bundles_root}")


if __name__ == "__main__":
    main()
