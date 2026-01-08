import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from model import PlantV5TripleBranch


def center_crop(img: Image.Image, crop_size: int) -> Image.Image:
    """Center crop to crop_size x crop_size."""
    w, h = img.size
    if w < crop_size or h < crop_size:
        side = min(w, h)
        left = (w - side) / 2
        top = (h - side) / 2
        return img.crop((left, top, left + side, top + side))
    left = (w - crop_size) / 2
    top = (h - crop_size) / 2
    return img.crop((left, top, left + crop_size, top + crop_size))


def load_image_pair(
    rgb_path: str,
    depth_path: str,
    image_size: int = 128,
    device: str = 'cuda',
    do_center_crop: bool = False,
    crop_size: int = 900,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load RGB/Depth, optional center-crop, then resize and normalize."""
    rgb = Image.open(rgb_path).convert('RGB')
    depth = Image.open(depth_path).convert('L')

    if do_center_crop:
        rgb = center_crop(rgb, crop_size)
        depth = center_crop(depth, crop_size)

    if rgb.size != (image_size, image_size):
        rgb = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if depth.size != (image_size, image_size):
        depth = depth.resize((image_size, image_size), Image.Resampling.BILINEAR)

    rgb_arr = np.array(rgb, dtype=np.float32) / 255.0
    depth_arr = np.array(depth, dtype=np.float32) / 255.0

    rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).to(device)  # (3, H, W)
    depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0).to(device)  # (1, H, W)
    rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # (4, H, W)

    return rgb_tensor, rgbd_tensor, depth_tensor


def predict_single_sample(
    model: torch.nn.Module,
    rgb: torch.Tensor,
    rgbd: torch.Tensor,
    depth: torch.Tensor,
    device: str = 'cuda',
) -> float:
    """Predict dry weight for a single sample.
    
    Args:
        rgb: (3, H, W) or (1, 3, H, W)
        rgbd: (4, H, W) or (1, 4, H, W)
        depth: (1, H, W) or (1, 1, H, W)
    
    Returns:
        prediction: scalar (dry weight)
    """
    model.eval()
    
    # Add batch dimension if needed
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    if rgbd.dim() == 3:
        rgbd = rgbd.unsqueeze(0)
    if depth.dim() == 3:
        depth = depth.unsqueeze(0)
    
    with torch.no_grad():
        pred = model(rgb, rgbd, depth)
    
    return pred.item() if pred.numel() == 1 else pred


def predict_batch(
    model: torch.nn.Module,
    rgb_batch: torch.Tensor,
    rgbd_batch: torch.Tensor,
    depth_batch: torch.Tensor,
    device: str = 'cuda',
) -> np.ndarray:
    """Predict dry weights for a batch.
    
    Args:
        rgb_batch: (N, 3, H, W)
        rgbd_batch: (N, 4, H, W)
        depth_batch: (N, 1, H, W)
    
    Returns:
        predictions: (N,) array
    """
    model.eval()
    
    with torch.no_grad():
        preds = model(rgb_batch, rgbd_batch, depth_batch)
    
    return preds.cpu().numpy()


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


def load_model(model_path: str, device: str) -> torch.nn.Module:
    """Load PlantV5TripleBranch with dims inferred from checkpoint."""
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
        print(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading checkpoint: {unexpected}")

    return model.to(device)


def predict_from_files(
    model: torch.nn.Module,
    rgb_dir: str,
    depth_dir: str,
    image_ids: list,
    device: str = 'cuda',
    image_size: int = 128,
    do_center_crop: bool = False,
    crop_size: int = 900,
) -> np.ndarray:
    """Predict for multiple images from file paths.
    
    Args:
        model: Trained model
        rgb_dir: Directory containing RGB_<id>.png
        depth_dir: Directory containing Depth_<id>.png
        image_ids: List of image IDs to predict on
        device: Device to use
    
    Returns:
        predictions: (N,) array
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for image_id in image_ids:
            rgb_path = os.path.join(rgb_dir, f"RGB_{image_id}.png")
            depth_path = os.path.join(depth_dir, f"Depth_{image_id}.png")
            
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"Warning: Missing files for {image_id}")
                predictions.append(np.nan)
                continue
            
            try:
                rgb, rgbd, depth = load_image_pair(
                    rgb_path,
                    depth_path,
                    image_size=image_size,
                    device=device,
                    do_center_crop=do_center_crop,
                    crop_size=crop_size,
                )
                
                # Add batch dimension
                rgb = rgb.unsqueeze(0)
                rgbd = rgbd.unsqueeze(0)
                depth = depth.unsqueeze(0)
                
                pred = model(rgb, rgbd, depth)
                predictions.append(pred.item())
            
            except Exception as e:
                print(f"Error predicting {image_id}: {e}")
                predictions.append(np.nan)
    
    return np.array(predictions)


def main(
    model_path: str,
    output_csv: str,
    rgb_dir: str,
    depth_dir: str,
    input_csv: str = None,
    device: str = 'cuda',
    image_size: int = 128,
    do_center_crop: bool = False,
    crop_size: int = 900,
):
    """Generate predictions and save to CSV.
    
    Args:
        model_path: Path to trained model checkpoint
        output_csv: Path to save predictions CSV
        rgb_dir: Directory containing RGB images
        depth_dir: Directory containing Depth images
        input_csv: Optional CSV with image_ids to predict on
        device: Device to use
    """
    # Load model
    actual_device = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    if actual_device != device:
        print("CUDA not available, falling back to CPU")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, actual_device)
    print(f"Model loaded successfully with device={actual_device}")
    
    # Get image IDs
    if input_csv:
        print(f"Loading image IDs from {input_csv}...")
        df = pd.read_csv(input_csv)
        id_col = 'image_id' if 'image_id' in df.columns else 'id'
        image_ids = df[id_col].tolist()
    else:
        print(f"Finding images in {rgb_dir}...")
        import glob
        rgb_files = glob.glob(os.path.join(rgb_dir, "RGB_*.png"))
        image_ids = [os.path.basename(f).replace("RGB_", "").replace(".png", "") for f in rgb_files]
    
    print(f"Predicting on {len(image_ids)} images...")
    predictions = predict_from_files(
        model,
        rgb_dir,
        depth_dir,
        image_ids,
        device=actual_device,
        image_size=image_size,
        do_center_crop=do_center_crop,
        crop_size=crop_size,
    )
    
    # Save predictions
    print(f"Saving predictions to {output_csv}...")
    output_df = pd.DataFrame({
        'image_id': image_ids,
        'DryWeightShoot': predictions,
    })
    output_df.to_csv(output_csv, index=False)
    print(f"Done! Saved {len(output_df)} predictions")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict using Model V5')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--rgb_dir', type=str, required=True, help='RGB images directory')
    parser.add_argument('--depth_dir', type=str, required=True, help='Depth images directory')
    parser.add_argument('--input_csv', type=str, default=None, help='Input CSV with image IDs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_size', type=int, default=128, help='Resize final image to this size')
    parser.add_argument('--center_crop', action='store_true', help='Center crop to 900x900 before resize')
    parser.add_argument('--crop_size', type=int, default=900, help='Center crop size if enabled')
    
    args = parser.parse_args()
    
    main(
        model_path=args.model,
        output_csv=args.output,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        input_csv=args.input_csv,
        device=args.device,
        image_size=args.image_size,
        do_center_crop=args.center_crop,
        crop_size=args.crop_size,
    )
