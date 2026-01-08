import logging
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import PlantDatasetV5
from model import PlantV5TripleBranch


def setup_logging(out_dir: str) -> logging.Logger:
    """Setup simple logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def _infer_dims_from_state(state_dict: dict) -> Tuple[Optional[int], Optional[int]]:
    """Infer branch_dim and fc_hidden from checkpoint shapes."""
    branch_dim: Optional[int] = None
    fc_hidden: Optional[int] = None

    first_fc_key = 'fusion_fc.net.0.weight'
    if first_fc_key in state_dict:
        fc_hidden = state_dict[first_fc_key].shape[0]
        in_features = state_dict[first_fc_key].shape[1]
        branch_dim = in_features // 3

    if branch_dim is None:
        for k in (
            'rgb_branch.head.1.weight',
            'rgbd_branch.head.1.weight',
            'depth_branch.head.1.weight',
        ):
            if k in state_dict:
                branch_dim = state_dict[k].shape[0]
                break

    return branch_dim, fc_hidden


def load_model(model_path: str, device: str, logger: logging.Logger) -> torch.nn.Module:
    """Load model with dims inferred from checkpoint to avoid size mismatches."""
    state_dict = torch.load(model_path, map_location='cpu')
    branch_dim, fc_hidden = _infer_dims_from_state(state_dict)

    model = PlantV5TripleBranch(
        branch_dim=branch_dim or 32,
        fc_hidden=fc_hidden or 64,
        dropout=0.2,
    )

    load_result = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
    if missing:
        logger.info(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        logger.info(f"Unexpected keys when loading checkpoint: {unexpected}")

    model = model.to(device)
    return model


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    logger: logging.Logger,
):
    """Evaluate model and compute metrics.
    
    Returns:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
    """
    model.eval()
    predictions = []
    labels_list = []
    
    with torch.no_grad():
        for rgb, rgbd, depth, labels in dataloader:
            rgb = rgb.to(device)
            rgbd = rgbd.to(device)
            depth = depth.to(device)
            labels = labels.to(device)

            pred = model(rgb, rgbd, depth)
            
            predictions.append(pred.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels_list)
    
    mae = np.abs(predictions - labels).mean()
    mse = np.square(predictions - labels).mean()
    rmse = np.sqrt(mse)
    
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"Min prediction: {predictions.min():.4f}")
    logger.info(f"Max prediction: {predictions.max():.4f}")
    logger.info(f"Mean prediction: {predictions.mean():.4f}")
    logger.info(f"Std prediction: {predictions.std():.4f}")
    
    return mae, rmse, predictions, labels


def main(
    model_path: str,
    csv_path: str,
    rgb_dir: str,
    depth_dir: str,
    device: str = 'cuda',
    batch_size: int = 32,
    out_dir: str = '.',
):
    """Evaluate model on dataset."""
    logger = setup_logging(out_dir)
    actual_device = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    if actual_device != device:
        logger.info(f"CUDA not available, falling back to CPU")

    logger.info("Loading model...")
    model = load_model(model_path, actual_device, logger)
    logger.info(f"Loaded model from {model_path} with device={actual_device}")
    
    logger.info("Loading dataset...")
    dataset = PlantDatasetV5(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        labels_csv=csv_path,
        image_size=128,
        device=actual_device,
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info("Running evaluation...")
    mae, rmse, predictions, labels = evaluate(model, dataloader, actual_device, logger)
    
    return mae, rmse


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Model V5')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, default='../../datasets/Training/Augmented/Train_aug.csv')
    parser.add_argument('--rgb_dir', type=str, default='../../datasets/Training/Augmented/RGBImages')
    parser.add_argument('--depth_dir', type=str, default='../../datasets/Training/Augmented/DepthImages')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_dir', type=str, default='.')
    
    args = parser.parse_args()
    
    main(
        model_path=args.model,
        csv_path=args.csv,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        device=args.device,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
    )
