
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ModelV2
from dataloader import PlantDatasetV2
import matplotlib.pyplot as plt

# Example paths (update as needed)
TRAIN_CSV = '../../datasets/Training/Augmented/Train_aug.csv'
RGB_DIR = '../../datasets/Training/Augmented/RGBImages/'
DEPTH_DIR = '../../datasets/Training/Augmented/DepthImages/'
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3



# Dataset and DataLoader with validation split
from torch.utils.data import random_split
full_dataset = PlantDatasetV2(RGB_DIR, DEPTH_DIR, TRAIN_CSV)
val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Debug: print unique y_class values
all_classes = set()
for i in range(len(full_dataset)):
    _, _, y_class, _ = full_dataset[i]
    all_classes.add(int(y_class))
print('Unique y_class indices in dataset:', all_classes)


# Model, Loss, Optimizer
import os
model = ModelV2(num_classes=4)
model = model.cuda() if torch.cuda.is_available() else model
# Resume from best model if exists
if os.path.exists('best_model_v2.pth'):
    model.load_state_dict(torch.load('best_model_v2.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    print('Resumed training from existing best_model_v2.pth')
else:
    print('No existing best model found, starting fresh.')


criterion_reg = nn.MSELoss()
criterion_fusion = nn.MSELoss()
criterion_class = nn.CrossEntropyLoss()
criterion_leaf = nn.MSELoss()

# Choose optimizer: AdamW (recommended) or Adam
use_adamw = True
if use_adamw:
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    print('Using AdamW optimizer')
else:
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print('Using Adam optimizer')

# Learning rate scheduler: ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

best_val_mae = float('inf')
save_after = 20  # Save best model after this many epochs

# --- Tracking for all losses ---
train_maes, val_maes = [], []
train_loss_regs, train_loss_classes, train_loss_leafs, train_loss_fusions, train_total_losses = [], [], [], [], []
val_loss_regs, val_loss_classes, val_loss_leafs, val_loss_fusions, val_total_losses = [], [], [], [], []





for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    train_mae = 0.0
    n_train = 0
    loss_reg_sum = 0.0
    loss_class_sum = 0.0
    loss_leaf_sum = 0.0
    loss_fusion_sum = 0.0
    for x, y_reg, y_class, leaf_area in train_loader:
        x = x.cuda() if torch.cuda.is_available() else x
        y_reg = y_reg.cuda() if torch.cuda.is_available() else y_reg
        y_class = y_class.cuda() if torch.cuda.is_available() else y_class
        if leaf_area is not None:
            leaf_area = leaf_area.cuda() if torch.cuda.is_available() else leaf_area
        optimizer.zero_grad()
        reg_out, class_out, leaf_out, fusion_out = model(x)
        loss_reg = criterion_reg(reg_out.squeeze(), y_reg)
        loss_class = criterion_class(class_out, y_class)
        loss_fusion = torch.sqrt(criterion_fusion(fusion_out.squeeze(), y_reg))
        loss_leaf = criterion_leaf(leaf_out.squeeze(), leaf_area) if leaf_area is not None else 0
        loss = 0.5 * loss_reg + loss_class * 0.4 + loss_leaf * 0.4 + loss_fusion * 2
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        # Track MAE for train
        mae = torch.abs(fusion_out.squeeze() - y_reg).sum().item()
        train_mae += mae
        n_train += y_reg.size(0)
        # Track individual losses
        loss_reg_sum += loss_reg.item() * x.size(0)
        loss_class_sum += loss_class.item() * x.size(0)
        loss_leaf_sum += (loss_leaf.item() if isinstance(loss_leaf, torch.Tensor) else loss_leaf) * x.size(0)
        loss_fusion_sum += loss_fusion.item() * x.size(0)
    train_mae = train_mae / n_train if n_train > 0 else 0
    train_maes.append(train_mae)
    train_loss_regs.append(loss_reg_sum / n_train if n_train > 0 else 0)
    train_loss_classes.append(loss_class_sum / n_train if n_train > 0 else 0)
    train_loss_leafs.append(loss_leaf_sum / n_train if n_train > 0 else 0)
    train_loss_fusions.append(loss_fusion_sum / n_train if n_train > 0 else 0)
    train_total_losses.append(running_loss / n_train if n_train > 0 else 0)

    # Validation losses
    model.eval()
    val_mae = 0.0
    n_val = 0
    val_loss_reg_sum = 0.0
    val_loss_class_sum = 0.0
    val_loss_leaf_sum = 0.0
    val_loss_fusion_sum = 0.0
    val_total_loss_sum = 0.0
    with torch.no_grad():
        for x, y_reg, y_class, leaf_area in val_loader:
            x = x.cuda() if torch.cuda.is_available() else x
            y_reg = y_reg.cuda() if torch.cuda.is_available() else y_reg
            y_class = y_class.cuda() if torch.cuda.is_available() else y_class
            if leaf_area is not None:
                leaf_area = leaf_area.cuda() if torch.cuda.is_available() else leaf_area
            reg_out, class_out, leaf_out, fusion_out = model(x)
            loss_reg = criterion_reg(reg_out.squeeze(), y_reg)
            loss_class = criterion_class(class_out, y_class)
            loss_fusion = criterion_fusion(fusion_out.squeeze(), y_reg)
            loss_leaf = criterion_leaf(leaf_out.squeeze(), leaf_area) if leaf_area is not None else 0
            loss = 0.8 * loss_reg + loss_class * 0.4 + loss_leaf * 0.4 + loss_fusion * 2
            val_loss_reg_sum += loss_reg.item() * x.size(0)
            val_loss_class_sum += loss_class.item() * x.size(0)
            val_loss_leaf_sum += (loss_leaf.item() if isinstance(loss_leaf, torch.Tensor) else loss_leaf) * x.size(0)
            val_loss_fusion_sum += loss_fusion.item() * x.size(0)
            val_total_loss_sum += loss.item() * x.size(0)
            mae = torch.abs(fusion_out.squeeze() - y_reg).sum().item()
            val_mae += mae
            n_val += y_reg.size(0)
    val_mae = val_mae / n_val if n_val > 0 else 0
    val_maes.append(val_mae)
    val_loss_regs.append(val_loss_reg_sum / n_val if n_val > 0 else 0)
    val_loss_classes.append(val_loss_class_sum / n_val if n_val > 0 else 0)
    val_loss_leafs.append(val_loss_leaf_sum / n_val if n_val > 0 else 0)
    val_loss_fusions.append(val_loss_fusion_sum / n_val if n_val > 0 else 0)
    val_total_losses.append(val_total_loss_sum / n_val if n_val > 0 else 0)

    # Print current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}, Train Loss: {train_total_losses[-1]:.4f}")

    # Step the scheduler with validation MAE
    scheduler.step(val_mae)

    # Save best model after a few epochs based on validation MAE
    if epoch + 1 >= save_after and val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), 'best_model_v2.pth')
        print(f"Best model saved at epoch {epoch+1} with val MAE {val_mae:.4f}")




# Plot all loss graphs at the end, each in its own subplot
epochs = list(range(1, EPOCHS+1))
fig, axs = plt.subplots(5, 1, figsize=(10, 18))
# MAE
axs[0].plot(epochs, train_maes, label='Train MAE')
axs[0].plot(epochs, val_maes, label='Val MAE')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('MAE')
axs[0].set_title('Training vs Validation MAE')
axs[0].legend()
axs[0].grid(True)
# Regression Loss
axs[1].plot(epochs, train_loss_regs, label='Train Reg')
axs[1].plot(epochs, val_loss_regs, label='Val Reg')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].set_title('Regression Loss (MSE)')
axs[1].legend()
axs[1].grid(True)
# Class Loss
axs[2].plot(epochs, train_loss_classes, label='Train Class')
axs[2].plot(epochs, val_loss_classes, label='Val Class')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Loss')
axs[2].set_title('Class Loss (CrossEntropy)')
axs[2].legend()
axs[2].grid(True)
# Leaf Loss
axs[3].plot(epochs, train_loss_leafs, label='Train Leaf')
axs[3].plot(epochs, val_loss_leafs, label='Val Leaf')
axs[3].set_xlabel('Epoch')
axs[3].set_ylabel('Loss')
axs[3].set_title('Leaf Area Loss (MSE)')
axs[3].legend()
axs[3].grid(True)
# Fusion Loss
axs[4].plot(epochs, train_loss_fusions, label='Train Fusion')
axs[4].plot(epochs, val_loss_fusions, label='Val Fusion')
axs[4].set_xlabel('Epoch')
axs[4].set_ylabel('Loss')
axs[4].set_title('Fusion Loss (MSE)')
axs[4].legend()
axs[4].grid(True)

plt.tight_layout()
fig.savefig('mae_and_losses_summary.png')
plt.show()
