import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import PlantDatasetV4
from model import LettuceMultiBranchCNN

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class EvalConfig:
    csv_path: str = '../../datasets/Training/Train.csv'
    rgb_dir: str = '../../datasets/Training/RGBImages'
    depth_dir: str = '../../datasets/Training/DepthImages'

    checkpoint: str = 'best_model_v4.pth'
    batch_size: int = 128
    seed: int = 42
    plot_path: str = 'eval_predictions.png'


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

    ds = PlantDatasetV4(cfg.rgb_dir, cfg.depth_dir, cfg.csv_path, augment=False, seed=cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = LettuceMultiBranchCNN().to(device)
    state = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mae = nn.L1Loss(reduction='sum')
    total_abs = 0.0
    total_n = 0

    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch['rgb'].to(device)
            rgbd = batch['rgbd'].to(device)
            y = batch['dry_weight'].to(device)

            _, _, fusion_pred = model(rgb, rgbd)
            total_abs += mae(fusion_pred, y).item()
            total_n += y.size(0)

            preds.extend(fusion_pred.detach().cpu().numpy().ravel().tolist())
            targets.extend(y.detach().cpu().numpy().ravel().tolist())

    final_mae = total_abs / max(1, total_n)
    print(f"MAE (dry weight): {final_mae:.6f}")

    if plt is None:
        print('matplotlib not available; skipping prediction scatter plot.')
        return

    if not preds:
        print('No predictions collected; skipping plot.')
        return

    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    line_min = float(min(targets_arr.min(), preds_arr.min()))
    line_max = float(max(targets_arr.max(), preds_arr.max()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets_arr, preds_arr, s=12, alpha=0.6, label='Samples')
    ax.plot([line_min, line_max], [line_min, line_max], color='tab:red', linestyle='--', label='Ideal')
    ax.set_xlabel('Ground truth DryWeightShoot')
    ax.set_ylabel('Predicted DryWeightShoot')
    ax.set_title(f'Dry-weight predictions (MAE={final_mae:.4f})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = Path(cfg.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Prediction scatter saved to: {plot_path}')


if __name__ == '__main__':
    main()
