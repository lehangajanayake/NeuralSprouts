import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloader import TestPlantDatasetV9
from model import LettuceNormalFusionNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_RGB = '../../datasets/Test/RGBImages'
TEST_DEPTH = '../../datasets/Test/DepthImages'
TEST_NORMAL = '../../datasets/Test/NormalMaps'
TEST_CSV = '../../datasets/Test/Test.csv'
MODEL_PATH = 'best_model_v9.pth'
IMAGE_SIZE = 96
OUTPUT_CSV = 'Test_with_predictions_v9.csv'


def predict_and_save(
    test_rgb: str = TEST_RGB,
    test_depth: str = TEST_DEPTH,
    test_normal: str = TEST_NORMAL,
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

    test_dataset = TestPlantDatasetV9(
        rgb_dir=test_rgb,
        depth_dir=test_depth,
        normal_dir=test_normal,
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

    model = LettuceNormalFusionNet().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    predictions = {}
    with torch.no_grad():
        for batch in test_loader:
            rgbn = batch['rgbn'].to(DEVICE, non_blocking=True)
            rgbd = batch['rgbd'].to(DEVICE, non_blocking=True)
            ids = batch['id']
            if isinstance(ids, torch.Tensor):
                ids = ids.cpu().numpy().tolist()

            preds = model.predict_dry_weight(rgbn, rgbd)
            preds = preds.detach().cpu().numpy().flatten().tolist()

            for image_id, pred in zip(ids, preds):
                predictions[int(image_id)] = float(pred)

    df = pd.read_csv(test_csv)
    if 'image_id' in df.columns:
        df['DryWeightShoot'] = df['image_id'].map(lambda x: predictions.get(int(x), ''))
    elif 'id' in df.columns:
        df['DryWeightShoot'] = df['id'].map(lambda x: predictions.get(int(x), ''))
    else:
        raise KeyError('Test CSV must have an image_id or id column')

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')


if __name__ == '__main__':
    predict_and_save()
