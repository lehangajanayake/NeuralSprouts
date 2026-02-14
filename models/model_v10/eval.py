"""Evaluate a model_v10 checkpoint on the original (un-augmented) training set.

Outputs:
* Per-sample CSV with predictions and absolute errors.
* Scatter plot of predicted vs ground-truth dry weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _reproducibility import seed_everything
from dataloader import PlantDatasetV10
from model import LettuceSAMFusionNet

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


@dataclass
class EvalConfig:
    csv_path: str = "../../datasets/Training/Train.csv"
    rgb_dir: str = "../../datasets/Training/RGBImages"
    depth_dir: str = "../../datasets/Training/DepthImages"

    checkpoint: str = "best_model_v10.pth"
    batch_size: int = 128
    seed: int = 42
    center_crop: bool = True
    blacklist_ids: Tuple[int, ...] = (163,)

    plot_path: str = "eval_predictions_v10.png"
    errors_csv: str = "eval_predictions_v10.csv"


def main(cfg: Optional[EvalConfig] = None) -> None:
    cfg = cfg or EvalConfig()
    seed_everything(cfg.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PlantDatasetV10(
        cfg.rgb_dir, cfg.depth_dir, cfg.csv_path,
        center_crop=cfg.center_crop, blacklist_ids=cfg.blacklist_ids,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = LettuceSAMFusionNet.from_checkpoint(cfg.checkpoint, device=device)
    model.eval()

    mae_fn = nn.L1Loss(reduction="sum")
    total_abs, total_n = 0.0, 0
    preds: List[float] = []
    targets: List[float] = []
    ids: List[int] = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            rgbd = batch["rgbd"].to(device)
            y = batch["dry_weight"].to(device)
            sid = batch["id"]
            _, _, fp = model(rgb, rgbd)
            # De-standardize predictions back to grams
            fp = model.destandardize(fp)
            total_abs += mae_fn(fp, y).item()
            total_n += y.size(0)
            preds.extend(fp.cpu().numpy().ravel().tolist())
            targets.extend(y.cpu().numpy().ravel().tolist())
            ids.extend(sid.cpu().numpy().ravel().astype(int).tolist())

    final_mae = total_abs / max(1, total_n)
    print(f"MAE (dry weight): {final_mae:.6f}")

    if not preds:
        return

    pa, ta = np.asarray(preds), np.asarray(targets)
    df = pd.DataFrame({"id": ids, "target": ta, "prediction": pa, "abs_error": np.abs(pa - ta)})
    df.sort_values("abs_error", ascending=False, inplace=True)
    ep = Path(cfg.errors_csv)
    ep.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ep, index=False)
    print(f"Per-sample predictions → {ep}")

    if plt is None:
        return
    lo = float(min(ta.min(), pa.min()))
    hi = float(max(ta.max(), pa.max()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ta, pa, s=12, alpha=0.6, label="Samples")
    ax.plot([lo, hi], [lo, hi], "r--", label="Ideal")
    ax.set(xlabel="Ground truth", ylabel="Predicted", title=f"Dry-weight  (MAE={final_mae:.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    pp = Path(cfg.plot_path)
    pp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot → {pp}")


if __name__ == "__main__":
    main()
