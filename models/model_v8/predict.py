import os
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloader import TestPlantDatasetV8
from model import LettuceSAMFusionNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_RGB = '../../datasets/Test/RGBImages'
TEST_DEPTH = '../../datasets/Test/DepthImages'
TEST_CSV = '../../datasets/Test/Test.csv'
MODEL_PATH = 'best_model_v8.pth'
IMAGE_SIZE = 96
OUTPUT_CSV = 'Test_with_predictions_v8.csv'
DROP_PATH_PROB = 0.1


def _infer_branch_widths(state_dict, branch_prefix: str) -> Tuple[int, ...]:
    widths = []
    idx = 0
    while True:
        key = f'{branch_prefix}.features.{idx}.conv3.1.weight'
        tensor = state_dict.get(key)
        if tensor is None:
            break
        widths.append(int(tensor.shape[0]))
        idx += 1
    if not widths:
        raise ValueError(f'Unable to infer widths for {branch_prefix}; checkpoint missing expected keys.')
    return tuple(widths)


def predict_and_save(
    test_rgb: str = TEST_RGB,
    test_depth: str = TEST_DEPTH,
    test_csv: str = TEST_CSV,
    model_path: str = MODEL_PATH,
    image_size: int = IMAGE_SIZE,
    output_csv: str = OUTPUT_CSV,
    batch_size: int = 64,
    blacklist_ids: Optional[Tuple[int, ...]] = (163,),
    drop_path_prob: float = DROP_PATH_PROB,
):
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f'Test CSV not found: {test_csv}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

    test_dataset = TestPlantDatasetV8(
        rgb_dir=test_rgb,
        depth_dir=test_depth,
        csv_file=test_csv,
        image_size=image_size,
        blacklist_ids=blacklist_ids,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 2,
        pin_memory=torch.cuda.is_available(),
    )

    state = torch.load(model_path, map_location=DEVICE)
    try:
        rgb_widths = _infer_branch_widths(state, 'rgb_branch')
        rgbd_widths = _infer_branch_widths(state, 'rgbd_branch')
        print(f"[predict] inferred widths rgb={rgb_widths} rgbd={rgbd_widths}")
    except ValueError as exc:
        print(f"[predict] {exc} — using default widths.")
        rgb_widths = (24, 48, 64, 96)
        rgbd_widths = (32, 64, 96, 128)

    model = LettuceSAMFusionNet(
        drop_path_prob=drop_path_prob,
        rgb_widths=rgb_widths,
        rgbd_widths=rgbd_widths,
    ).to(DEVICE)
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

            preds = model.predict_dry_weight(rgb, rgbd)
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

    df.to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')


if __name__ == '__main__':
    predict_and_save()
