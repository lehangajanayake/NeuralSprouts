import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloader import TestPlantDatasetV4
from model import LettuceMultiBranchCNN


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defaults (match repo layout)
TEST_RGB = '../../datasets/Test/RGBImages'
TEST_DEPTH = '../../datasets/Test/DepthImages'
TEST_CSV = '../../datasets/Test/Test.csv'
MODEL_PATH = 'best_model_v4.pth'
IMAGE_SIZE = 64
OUTPUT_CSV = 'Test_with_predictions_v4.csv'


def predict_and_save(
    test_rgb: str = TEST_RGB,
    test_depth: str = TEST_DEPTH,
    test_csv: str = TEST_CSV,
    model_path: str = MODEL_PATH,
    image_size: int = IMAGE_SIZE,
    output_csv: str = OUTPUT_CSV,
    batch_size: int = 64,
):
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f'Test CSV not found: {test_csv}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

    test_dataset = TestPlantDatasetV4(
        rgb_dir=test_rgb,
        depth_dir=test_depth,
        csv_file=test_csv,
        image_size=image_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 2,
        pin_memory=torch.cuda.is_available(),
    )

    # num_classes isn't needed for inference output (we only submit dry weight),
    # but the module needs the right shape for the RGB head.
    # We keep 4 to match the competition setup.
    model = LettuceMultiBranchCNN(num_classes=4).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    predictions = {}
    with torch.no_grad():
        for batch in test_loader:
            rgb = batch['rgb'].to(DEVICE, non_blocking=True)
            rgbd = batch['rgbd'].to(DEVICE, non_blocking=True)
            ids = batch['id']
            if isinstance(ids, torch.Tensor):
                ids = ids.cpu().numpy().tolist()

            fusion_pred = model.predict_dry_weight(rgb, rgbd)
            preds = fusion_pred.detach().cpu().numpy().flatten().tolist()

            for image_id, pred in zip(ids, preds):
                predictions[int(image_id)] = float(pred)

    # Read original test CSV and overwrite DryWeightShoot
    df = pd.read_csv(test_csv)
    if 'image_id' in df.columns:
        df['DryWeightShoot'] = df['image_id'].map(lambda x: predictions.get(int(x), ''))
    elif 'id' in df.columns:
        df['DryWeightShoot'] = df['id'].map(lambda x: predictions.get(int(x), ''))
    else:
        raise KeyError('Test CSV must have an image_id or id column')

    df.to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')


if __name__ == '__main__':
    predict_and_save()
