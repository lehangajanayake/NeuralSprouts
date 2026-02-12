import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import PlantDatasetV4
from model import LettuceMultiBranchCNN


@dataclass
class EvalConfig:
    csv_path: str = '../../datasets/Training/Train.csv'
    rgb_dir: str = '../../datasets/Training/RGBImages'

    checkpoint: str = 'best_model_v4.pth'
    batch_size: int = 128
    seed: int = 42


def seed_everything(seed: int = 42, deterministic: bool = True):
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Be robust: some torch builds don't expose this API, and even when they do,
        # enabling strict determinism may raise depending on platform/backend.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def main(cfg: Optional[EvalConfig] = None):
    cfg = cfg or EvalConfig()
    seed_everything(cfg.seed, deterministic=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = PlantDatasetV4(cfg.rgb_dir, cfg.csv_path, augment=False, seed=cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = LettuceMultiBranchCNN().to(device)
    state = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mae = nn.L1Loss(reduction='sum')
    total_abs = 0.0
    total_n = 0

    with torch.no_grad():
        for batch in loader:
            rgb = batch['rgb'].to(device)
            y = batch['dry_weight'].to(device)

            preds = model(rgb)
            total_abs += mae(preds, y).item()
            total_n += y.size(0)

    final_mae = total_abs / max(1, total_n)
    print(f"MAE (dry weight): {final_mae:.6f}")


if __name__ == '__main__':
    main()
