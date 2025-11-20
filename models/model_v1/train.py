import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from model_v1.dataloader import LettuceDataset
from model_v1.model import model_v1

def train():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 10
    lr = 0.001
    image_size = 64
    # Dataset
    dataset = LettuceDataset(
        RGB_dir="../datasets/Training/RGBImages",
        depth_dir="../datasets/Training/DepthImages",
        labels_file="../datasets/Training/Train.csv",
        image_size=image_size
    )
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    # Model
    model = model_v1().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_mae += torch.abs(outputs - labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    print("Model v1 architecture:")
    model = model_v1()
    print(model)

    print("\nStarting training...")
    train()