
import torch
import matplotlib.pyplot as plt
from model_v1.dataloader import LettuceDataset
from model_v1.model import model_v1
from model_v1.augmented_dataloader import AugmentedLettuceDataset

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_RGB = '../datasets/Training/RGBImages'
TRAIN_DEPTH = '../datasets/Training/DepthImages'
TRAIN_CSV = '../datasets/Training/Train.csv'
MODEL_PATH = 'best_model.pth'
IMAGE_SIZE = 64

def evaluate_on_training():
    print('Loading training dataset...')
    train_dataset = LettuceDataset(
        RGB_dir=TRAIN_RGB,
        depth_dir=TRAIN_DEPTH,
        labels_file=TRAIN_CSV,
        image_size=IMAGE_SIZE
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)

    print('Loading model...')
    num_classes = len(train_dataset.variety2idx)
    model = model_v1(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []
    image_ids = []
    with torch.no_grad():
        for images, dry_weight, variety_class, image_id in train_loader:
            images = images.to(DEVICE)
            dry_weight = dry_weight.to(DEVICE)
            # Get only the fusion output
            _, _, fusion_out = model(images)
            y_true.extend(dry_weight.cpu().numpy().flatten().tolist())
            y_pred.extend(fusion_out.cpu().numpy().flatten().tolist())
            # image_id is a batch of IDs
            if isinstance(image_id, torch.Tensor):
                image_ids.extend(image_id.cpu().numpy().tolist())
            else:
                image_ids.extend(list(image_id))

    # Print MAE
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"Mean Absolute Error on Training Data: {mae:.4f}")

    # Print a table of predictions with image IDs
    print("Sample predictions (image_id, actual, predicted):")
    for i in range(min(10, len(image_ids))):
        print(f"{image_ids[i]}: {y_true[i]:.4f} -> {y_pred[i]:.4f}")
        if i == 0:
            errors = [abs(t - p) for t, p in zip(y_true, y_pred)]
            top_n = min(10, len(errors))
            top = sorted(enumerate(errors), key=lambda x: x[1], reverse=True)[:top_n]
            print("\nMost erroneous data (rank, image_id, actual, predicted, abs_error):")
            for rank, (idx, err) in enumerate(top, start=1):
                img_id = image_ids[idx] if idx < len(image_ids) else f"idx_{idx}"
                print(f"{rank}. {img_id}: {y_true[idx]:.4f} -> {y_pred[idx]:.4f}, error={err:.4f}")

    # Plot actual vs predicted (fusion)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal')
    plt.xlabel('Actual Dry Weight')
    plt.ylabel('Fusion Predicted Dry Weight')
    plt.title('Actual vs Fusion Predicted Dry Weight (Training Data)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_on_training()
