import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataloader import SimplePlantDataset
from model import SimpleResNetModel


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defaults (match repo layout)
TEST_RGB = '../../datasets/Test/RGBImages'
TEST_CSV = '../../datasets/Test/Test.csv'
MODEL_PATH = 'best_model_v3.pth'
IMAGE_SIZE = 224
OUTPUT_CSV = 'Test_with_predictions_v3.csv'


def predict_and_save(
    test_rgb: str = TEST_RGB,
    test_csv: str = TEST_CSV,
    model_path: str = MODEL_PATH,
    image_size: int = IMAGE_SIZE,
    output_csv: str = OUTPUT_CSV,
    batch_size: int = 32,
):
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f'Test CSV not found: {test_csv}')

    # Use the same dataset class (it just needs id + image existence). We don't
    # care about the blank DryWeightShoot values in Test.csv.
    test_dataset = SimplePlantDataset(test_rgb, test_csv, image_size=image_size)
    test_dataset.num_views = 1  # no augmentation / multi-view for prediction
    test_dataset.enable_cache = False

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 2,
        pin_memory=torch.cuda.is_available(),
    )

    model = SimpleResNetModel().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    preds = {}
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE, non_blocking=True)
            y_hat = model(x).squeeze(1)
            y_hat = y_hat.detach().cpu().numpy().tolist()

            # Map back to ids in the same order as the dataset
            # (dataset.df is already filtered for existing images)
            # We can use the underlying dataframe indices for this batch.
            # DataLoader yields samples in order when shuffle=False.
            # We track a running offset into dataset.df.
            #
            # Note: dataset returns N samples where N == len(dataset.df), since num_views=1.
            #
            # We'll fill preds using dataset.df['id'].
            if not hasattr(test_loader, '_v3_offset'):
                test_loader._v3_offset = 0
            offset = test_loader._v3_offset
            ids = test_dataset.df.iloc[offset:offset + len(y_hat)]['id'].tolist()
            test_loader._v3_offset += len(y_hat)

            for img_id, pred in zip(ids, y_hat):
                preds[int(img_id)] = float(pred)

    # Write output CSV: copy original, overwrite DryWeightShoot
    df = pd.read_csv(test_csv)
    if 'image_id' in df.columns:
        df['DryWeightShoot'] = df['image_id'].map(lambda x: preds.get(int(x), ''))
    elif 'id' in df.columns:
        df['DryWeightShoot'] = df['id'].map(lambda x: preds.get(int(x), ''))
    else:
        raise KeyError('Test CSV must have an image_id or id column')

    df.to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')


if __name__ == '__main__':
    predict_and_save()
