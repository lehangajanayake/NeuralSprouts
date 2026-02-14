"""Generate test-set predictions for model_v10.

Reads the test PNGs (small set — no sharding needed), runs inference, and
writes a CSV with ``DryWeightShoot`` predictions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from _reproducibility import seed_everything
from dataloader import TestPlantDataset
from model import LettuceSAMFusionNet


@dataclass
class PredictConfig:
    test_rgb: str = "../../datasets/Test/RGBImages"
    test_depth: str = "../../datasets/Test/DepthImages"
    test_csv: str = "../../datasets/Test/Test.csv"
    checkpoint: str = "best_model_v10.pth"
    image_size: int = 128
    output_csv: str = "Test_with_predictions_v10.csv"
    batch_size: int = 64
    blacklist_ids: Tuple[int, ...] = (163,)
    seed: int = 42


def predict_and_save(cfg: Optional[PredictConfig] = None) -> None:
    cfg = cfg or PredictConfig()
    seed_everything(cfg.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(cfg.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {cfg.test_csv}")
    if not os.path.exists(cfg.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")

    ds = TestPlantDataset(
        cfg.test_rgb, cfg.test_depth, cfg.test_csv,
        image_size=cfg.image_size, blacklist_ids=cfg.blacklist_ids,
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
        pin_memory=torch.cuda.is_available(),
    )

    model = LettuceSAMFusionNet.from_checkpoint(cfg.checkpoint, device=device)
    model.eval()

    predictions: dict[int, float] = {}
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            rgbd = batch["rgbd"].to(device, non_blocking=True)
            ids = batch["id"]
            if isinstance(ids, torch.Tensor):
                ids = ids.cpu().numpy().tolist()
            preds = model.predict_dry_weight(rgb, rgbd).cpu().numpy().flatten().tolist()
            for iid, p in zip(ids, preds):
                predictions[int(iid)] = float(p)

    df = pd.read_csv(cfg.test_csv)
    id_col = "image_id" if "image_id" in df.columns else "id"
    df["DryWeightShoot"] = df[id_col].map(lambda x: predictions.get(int(x), ""))
    df.to_csv(cfg.output_csv, index=False)
    print(f"Predictions saved → {cfg.output_csv}")


if __name__ == "__main__":
    predict_and_save()
