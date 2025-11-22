import torch
from torch.utils.data import DataLoader
from model import ModelV2
from dataloader import PlantDatasetV2

# Example paths (update as needed)

TEST_CSV = '../../datasets/Training/Train.csv'
RGB_DIR = '../../datasets/Training/RGBImages/'
DEPTH_DIR = '../../datasets/Training/DepthImages/'

BATCH_SIZE = 16

# Dataset and DataLoader
test_dataset = PlantDatasetV2(TEST_CSV, RGB_DIR, DEPTH_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = ModelV2(num_classes=3)
model.load_state_dict(torch.load('best_model_v2.pth', map_location='cpu'))
model.eval()
model = model.cuda() if torch.cuda.is_available() else model

results = []
with torch.no_grad():
    for x, y_reg, y_class, leaf_area in test_loader:
        x = x.cuda() if torch.cuda.is_available() else x
        reg_out, class_out, leaf_out, fusion_out = model(x)
        results.append({
            'regression': reg_out.cpu().numpy(),
            'classification': class_out.cpu().numpy(),
            'leaf_area': leaf_out.cpu().numpy(),
            'fusion': fusion_out.cpu().numpy()
        })
# Post-process and save results as needed
