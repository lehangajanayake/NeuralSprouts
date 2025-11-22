import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from model_v1.augmented_dataloader import AugmentedLettuceDataset
from model_v1.model import model_v1

import os

def train():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 200
    lr = 0.001
    image_size = 64
    # Dataset
    dataset = AugmentedLettuceDataset(
        RGB_dir="../datasets/Training/Augmented/RGBImages",
        depth_dir="../datasets/Training/Augmented/DepthImages",
        labels_file="../datasets/Training/Augmented/Train_aug.csv",
        image_size=image_size
    )
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    # Model

    # Get number of classes from dataset
    num_classes = len(dataset.variety2idx)
    model = model_v1(num_classes=num_classes).to(device)


    # Load previous best weights if available
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("Loaded previous best_model.pth weights.")
    criterion_reg = nn.MSELoss()
    criterion_fusion = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Learning rate scheduler: Reduce LR if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


    # Training loop
    import matplotlib.pyplot as plt
    train_losses = []
    val_losses = []
    val_maes = []
    val_cls_accs = []
    best_val_mae = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, dry_weight, variety_class, _ in train_loader:
            images = images.to(device)
            dry_weight = dry_weight.to(device)
            variety_class = variety_class.to(device)
            optimizer.zero_grad()
            reg_out, cls_out, fusion_out = model(images)
            loss_reg = criterion_reg(reg_out.squeeze(), dry_weight)
            loss_cls = criterion_cls(cls_out, variety_class)
            loss_fusion = criterion_fusion(fusion_out.squeeze(), dry_weight)
            # Combine losses (weights can be tuned)
            loss = loss_reg + 0.5 * loss_cls + 2 *loss_fusion
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Learning Rate = {current_lr:.6f}")


        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_cls_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for images, dry_weight, variety_class, _ in val_loader:
                images = images.to(device)
                dry_weight = dry_weight.to(device)
                variety_class = variety_class.to(device)
                reg_out, cls_out, fusion_out = model(images)


                loss_reg = criterion_reg(reg_out.squeeze(), dry_weight)
                loss_cls = criterion_cls(cls_out, variety_class)
                loss_fusion = criterion_fusion(fusion_out.squeeze(), dry_weight)
                loss = loss_reg + 0.5 * loss_cls + 2 *loss_fusion


                val_loss += loss.item() * images.size(0)
                val_mae += torch.abs(fusion_out.squeeze() - dry_weight).sum().item()


                # Classification accuracy
                preds = torch.argmax(cls_out, dim=1)
                val_cls_acc += (preds == variety_class).sum().item()
                n_val += images.size(0)
        val_loss /= n_val
        val_mae /= n_val
        val_cls_acc /= n_val
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val Cls Acc: {val_cls_acc:.4f}")

        # Save metrics for plotting
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_cls_accs.append(val_cls_acc)

        # Step the scheduler on validation loss
        scheduler.step(val_loss)

        # Save best model by val MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Saved new best model (Val MAE: {val_mae:.4f})")
    print("Training complete.")

    # Plot training summary
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_maes, label='Val MAE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_cls_accs, label='Val Cls Acc', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Classification Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Model v1 architecture:")
    model = model_v1()
    print(model)

    print("\nStarting training...")
    train()