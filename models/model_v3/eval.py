import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader import SimplePlantDataset
from model import SimpleResNetModel


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defaults (update as needed)
EVAL_CSV = '../../datasets/Training/Train.csv'
RGB_DIR = '../../datasets/Training/RGBImages/'
MODEL_PATH = 'best_model_v3.pth'
IMAGE_SIZE = 224
BATCH_SIZE = 32


def eval_model(
    rgb_dir: str = RGB_DIR,
    eval_csv: str = EVAL_CSV,
    model_path: str = MODEL_PATH,
    image_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
):
    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f'Eval CSV not found: {eval_csv}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

    dataset = SimplePlantDataset(rgb_dir, eval_csv, image_size=image_size)
    dataset.num_views = 1
    dataset.enable_cache = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 2,
        pin_memory=torch.cuda.is_available(),
    )

    model = SimpleResNetModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds: list[float] = []
    all_targets: list[float] = []
    all_ids: list[int] = []

    with torch.no_grad():
        offset = 0
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            y_hat = model(x).squeeze(1)

            preds = y_hat.detach().cpu().numpy().astype(np.float32)
            targets = y.detach().cpu().numpy().astype(np.float32)

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

            ids = dataset.df.iloc[offset:offset + len(preds)]['id'].tolist()
            all_ids.extend([int(v) for v in ids])
            offset += len(preds)

    all_preds_np = np.asarray(all_preds, dtype=np.float32)
    all_targets_np = np.asarray(all_targets, dtype=np.float32)
    abs_errors = np.abs(all_preds_np - all_targets_np)

    mae = float(abs_errors.mean()) if len(abs_errors) else float('nan')
    print(f"Eval MAE: {mae:.4f}")

    # Plot regression: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets_np, all_preds_np, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Regression: Actual vs Predicted')
    if len(all_targets_np):
        lo = float(min(all_targets_np.min(), all_preds_np.min()))
        hi = float(max(all_targets_np.max(), all_preds_np.max()))
        plt.plot([lo, hi], [lo, hi], 'r--')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('v3_eval_actual_vs_pred.png')
    plt.show()

    # Plot error histogram
    plt.figure(figsize=(6, 4))
    plt.hist(abs_errors, bins=30, alpha=0.7)
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.title('Histogram of Absolute Errors')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('v3_eval_abs_error_hist.png')
    plt.show()

    # Print worst-performing images
    if all_ids:
        tuples = list(zip(all_ids, all_targets_np.tolist(), all_preds_np.tolist(), abs_errors.tolist()))
        tuples.sort(key=lambda t: -t[3])
        print('\nWorst-performing images (top 10 by absolute error):')
        for i, (img_id, target, pred, err) in enumerate(tuples[:10]):
            print(f"{i+1}. ID: {img_id} | Actual: {target:.2f} | Predicted: {pred:.2f} | Abs Error: {err:.2f}")

    print('Saved plots: v3_eval_actual_vs_pred.png, v3_eval_abs_error_hist.png')


if __name__ == '__main__':
    eval_model()
