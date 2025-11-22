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
test_dataset = PlantDatasetV2(RGB_DIR, DEPTH_DIR, TEST_CSV)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = ModelV2(num_classes=4)
model.load_state_dict(torch.load('best_model_v2.pth', map_location='cpu'))
model.eval()
model = model.cuda() if torch.cuda.is_available() else model


# Collect predictions, targets, and image ids
import matplotlib.pyplot as plt
all_preds = []
all_targets = []
all_image_ids = []
abs_errors = []

with torch.no_grad():
    for x, y_reg, y_class, leaf_area in test_loader:
        x = x.cuda() if torch.cuda.is_available() else x
        reg_out, class_out, leaf_out, fusion_out = model(x)
        preds = reg_out.cpu().numpy().flatten()
        targets = y_reg.cpu().numpy().flatten()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
        # Try to get image ids if available
        if hasattr(test_loader.dataset, 'df') and 'id' in test_loader.dataset.df.columns:
            batch_indices = test_loader.batch_size if hasattr(test_loader, 'batch_size') else len(targets)
            start_idx = len(all_image_ids)
            for i in range(len(targets)):
                all_image_ids.append(str(test_loader.dataset.df.iloc[start_idx + i]['id']))
        abs_errors.extend(abs(preds - targets).tolist())

# Compute MAE
import numpy as np
mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
print(f"Test MAE: {mae:.4f}")

# Plot regression: Actual vs Predicted
plt.figure()
plt.scatter(all_targets, all_preds, alpha=0.5)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Regression: Actual vs Predicted')
plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
plt.savefig('test_regression_actual_vs_pred.png')
plt.show()

# Plot error histogram
plt.figure()
plt.hist(abs_errors, bins=30, alpha=0.7)
plt.xlabel('Absolute Error')
plt.ylabel('Count')
plt.title('Histogram of Absolute Errors')
plt.savefig('test_abs_error_hist.png')
plt.show()

# Print worst-performing images (top 10 by absolute error)
if all_image_ids:
    error_tuples = list(zip(all_image_ids, all_targets, all_preds, abs_errors))
    error_tuples.sort(key=lambda x: -x[3])
    print("\nWorst-performing images (top 10 by absolute error):")
    for i, (img_id, target, pred, err) in enumerate(error_tuples[:10]):
        print(f"{i+1}. ID: {img_id} | Actual: {target:.2f} | Predicted: {pred:.2f} | Abs Error: {err:.2f}")
