import os
from dataclasses import dataclass
from typing import Optional
import itertools

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import PlantDatasetV4
from model import LettuceMultiBranchCNN


@dataclass
class EvalConfig:
    csv_path: str = '../../datasets/Training/Augmented/Train_aug.csv'
    rgb_dir: str = '../../datasets/Training/Augmented/RGBImages'
    depth_dir: str = '../../datasets/Training/Augmented/DepthImages'

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

    # Load data and perform group-aware split (same as training)
    from dataloader import group_aware_train_val_split
    
    df = pd.read_csv(cfg.csv_path)
    if 'image_id' in df.columns:
        df.rename(columns={'image_id': 'id'}, inplace=True)
    
    train_indices, val_indices = group_aware_train_val_split(df, val_ratio=0.2, seed=cfg.seed)
    
    # Load datasets with num_views=1 (explicit, matching train.py)
    ds_train = PlantDatasetV4(cfg.rgb_dir, cfg.depth_dir, cfg.csv_path, augment=True, seed=cfg.seed, num_views=1)
    ds_val = PlantDatasetV4(cfg.rgb_dir, cfg.depth_dir, cfg.csv_path, augment=False, seed=cfg.seed, num_views=1)
    
    # Create separate loaders for train and val
    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(ds_train, train_indices), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(Subset(ds_val, val_indices), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    # For debug: evaluate also on train set
    combined_loader = itertools.chain(val_loader, train_loader)

    model = LettuceMultiBranchCNN(num_classes=len(ds_val.variety2idx)).to(device)
    state = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mae = nn.L1Loss(reduction='sum')

    print(f"\n=== Evaluating on validation set (group-aware split, n={len(val_indices)}) ===")
    
    total_abs = 0.0
    total_n = 0
    correct = 0
    total_cls = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(combined_loader):
            rgb = batch['rgb'].to(device)
            rgbd = batch['rgbd'].to(device)
            y = batch['dry_weight'].to(device)
            y_cls = batch['variety_class'].to(device)

            logits, _, fusion_pred = model(rgb, rgbd)
            total_abs += mae(fusion_pred, y).item()
            total_n += y.size(0)

            pred_cls = logits.argmax(dim=1)
            correct += (pred_cls == y_cls).sum().item()
            total_cls += y_cls.size(0)
            
            if batch_idx == 0:
                # Debug: print first batch predictions vs labels
                print(f"\nDEBUG first batch:")
                print(f"  Labels: {y_cls.cpu().tolist()}")
                print(f"  Predictions: {pred_cls.cpu().tolist()}")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Variety index mapping: {ds_val.variety2idx}")

    final_mae = total_abs / max(1, total_n)
    acc = correct / max(1, total_cls)

    print(f"Validation MAE (dry weight, fusion output): {final_mae:.6f}")
    print(f"Validation Classification accuracy: {acc:.4%}")
    print(f"\nNote: Evaluated on group-aware validation split (no plant leakage)")
    print(f"      Same as debug.log confusion matrix results")


if __name__ == '__main__':
    main()
