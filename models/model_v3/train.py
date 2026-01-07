import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import SimpleResNetModel
from dataloader import SimplePlantDataset
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import random
import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True):
    """Seed Python/NumPy/PyTorch for (mostly) reproducible training.

    Notes:
    - Full determinism can reduce speed.
    - Some CUDA ops can still be non-deterministic depending on hardware.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def seed_worker(worker_id: int):
    # Make each worker's RNG deterministic but different.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def estimate_cache_ram_gb(num_base_images: int, num_views: int, image_size: int = 224, dtype_bytes: int = 4) -> float:
    """Rough RAM estimate for caching tensors: N * K * (3*H*W*dtype_bytes)."""
    bytes_per = 3 * image_size * image_size * dtype_bytes
    total_bytes = int(num_base_images) * int(num_views) * bytes_per
    return total_bytes / (1024**3)

def plot_and_save(train_maes, val_maes, train_losses, val_losses, out_path='summary_v3.png'):
    # Robust to early interrupt: plot only the available points.
    n = min(len(train_maes), len(val_maes))
    epochs = list(range(1, n + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_maes[:n], label='Train MAE')
    plt.plot(epochs, val_maes[:n], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses[:n], label='Train MSE')
    plt.plot(epochs, val_losses[:n], label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Regression Loss (MSE)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


def main():
    # Paths (update as needed)
    TRAIN_CSV = '../../datasets/Training/Train.csv'
    RGB_DIR = '../../datasets/Training/RGBImages/'
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    SEED = 42
    NUM_VIEWS = 4  # number of cached random augmentations per image
    CACHE_MAX_ITEMS = None  # set to an int to cache only a subset (safer on RAM)

    seed_everything(SEED, deterministic=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader with validation split
    full_dataset = SimplePlantDataset(RGB_DIR, TRAIN_CSV)
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Cache multiple random augmented "views" per image for faster epochs.
    # This trades RAM for speed and helps reduce overfitting.
    full_dataset.num_views = NUM_VIEWS
    full_dataset.cache_seed = SEED
    full_dataset.enable_cache = True
    est_gb = estimate_cache_ram_gb(len(full_dataset.df), NUM_VIEWS, image_size=224, dtype_bytes=4)
    print(f"[cache] Planning to cache {len(full_dataset.df)} images x {NUM_VIEWS} views (~{est_gb:.2f} GB RAM for float32 tensors).")
    if est_gb >= 8.0 and CACHE_MAX_ITEMS is None:
        print("[cache] WARNING: This may exceed your available RAM. Consider lowering NUM_VIEWS or setting CACHE_MAX_ITEMS.")

    full_dataset.build_cache(max_items=CACHE_MAX_ITEMS)

    # On Windows you MUST guard entrypoint when using num_workers>0.
    # Keep 0 as a safe default; you can increase once everything works.
    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = torch.cuda.is_available()
    # Generator makes shuffle order reproducible when combined with seeds.
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Model, Loss, Optimizer (regression only)
    model = SimpleResNetModel().to(DEVICE)

    criterion_reg = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    train_maes, val_maes = [], []
    train_losses, val_losses = [], []

    best_val_mae = float('inf')
    best_path = 'best_model_v3.pth'

    try:
        for epoch in range(EPOCHS):
            model.train()
            train_mae_sum = 0.0
            train_loss_sum = 0.0
            n_train = 0

            for x, y_reg in train_loader:
                x = x.to(DEVICE, non_blocking=True)
                y_reg = y_reg.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                reg_out = model(x)

                loss_reg = criterion_reg(reg_out.squeeze(), y_reg)
                loss = loss_reg

                loss.backward()
                optimizer.step()

                batch_size = y_reg.size(0)
                train_mae_sum += torch.abs(reg_out.squeeze() - y_reg).sum().item()
                train_loss_sum += loss_reg.item() * batch_size
                n_train += batch_size

            train_maes.append(train_mae_sum / n_train)
            train_losses.append(train_loss_sum / n_train)

            # Validation
            model.eval()
            val_mae_sum = 0.0
            val_loss_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for x, y_reg in val_loader:
                    x = x.to(DEVICE, non_blocking=True)
                    y_reg = y_reg.to(DEVICE, non_blocking=True)

                    reg_out = model(x)
                    loss_reg = criterion_reg(reg_out.squeeze(), y_reg)

                    batch_size = y_reg.size(0)
                    val_mae_sum += torch.abs(reg_out.squeeze() - y_reg).sum().item()
                    val_loss_sum += loss_reg.item() * batch_size
                    n_val += batch_size

            val_mae = val_mae_sum / n_val
            val_maes.append(val_mae)
            val_losses.append(val_loss_sum / n_val)

            scheduler.step(val_mae)
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | "
                f"Train MAE: {train_maes[-1]:.4f} | Val MAE: {val_mae:.4f}"
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), best_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving plots with progress so far...")
        plot_and_save(
            train_maes, val_maes,
            train_losses, val_losses,
            out_path='summary_v3.png'
        )
        print("Saved summary_v3.png")
        return

    plot_and_save(
        train_maes, val_maes,
        train_losses, val_losses,
        out_path='summary_v3.png'
    )
    print(f"Saved summary_v3.png and best model to {best_path} (best val MAE={best_val_mae:.4f})")


if __name__ == '__main__':
    mp.freeze_support()
    main()
