import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import SimpleResNetModel
from dataloader import SimplePlantDataset
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

def plot_and_save(train_maes, val_maes, train_class_accs, val_class_accs, train_reg_losses, val_reg_losses, train_class_losses, val_class_losses, out_path='summary_v3.png'):
    epochs = list(range(1, len(train_maes) + 1))
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_maes, label='Train MAE')
    plt.plot(epochs, val_maes, label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_class_accs, label='Train Acc')
    plt.plot(epochs, val_class_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_reg_losses, label='Train Reg Loss')
    plt.plot(epochs, val_reg_losses, label='Val Reg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Regression Loss (MSE)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_class_losses, label='Train Class Loss')
    plt.plot(epochs, val_class_losses, label='Val Class Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Classification Loss (CrossEntropy)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


def main():
    # Paths (update as needed)
    TRAIN_CSV = '../../datasets/Training/Train.csv'
    RGB_DIR = '../../datasets/Training/RGBImages/'
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader with validation split
    full_dataset = SimplePlantDataset(RGB_DIR, TRAIN_CSV)
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # On Windows you MUST guard entrypoint when using num_workers>0.
    # Keep 0 as a safe default; you can increase once everything works.
    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Model, Loss, Optimizer
    num_classes = len(full_dataset.variety2idx)
    model = SimpleResNetModel(num_classes=num_classes).to(DEVICE)

    criterion_reg = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    train_maes, val_maes = [], []
    train_class_accs, val_class_accs = [], []
    train_reg_losses, val_reg_losses = [], []
    train_class_losses, val_class_losses = [], []

    best_val_mae = float('inf')
    best_path = 'best_model_v3.pth'

    try:
        for epoch in range(EPOCHS):
            model.train()
            train_mae_sum = 0.0
            train_acc_sum = 0.0
            train_reg_loss_sum = 0.0
            train_class_loss_sum = 0.0
            n_train = 0

            for x, y_reg, y_class in train_loader:
                x = x.to(DEVICE, non_blocking=True)
                y_reg = y_reg.to(DEVICE, non_blocking=True)
                y_class = y_class.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                reg_out, class_out = model(x)

                loss_reg = criterion_reg(reg_out.squeeze(), y_reg)
                loss_class = criterion_class(class_out, y_class)
                loss = loss_reg + 0.3 * loss_class

                loss.backward()
                optimizer.step()

                batch_size = y_reg.size(0)
                train_mae_sum += torch.abs(reg_out.squeeze() - y_reg).sum().item()
                preds = class_out.argmax(dim=1)
                train_acc_sum += (preds == y_class).sum().item()
                train_reg_loss_sum += loss_reg.item() * batch_size
                train_class_loss_sum += loss_class.item() * batch_size
                n_train += batch_size

            train_maes.append(train_mae_sum / n_train)
            train_class_accs.append(train_acc_sum / n_train)
            train_reg_losses.append(train_reg_loss_sum / n_train)
            train_class_losses.append(train_class_loss_sum / n_train)

            # Validation
            model.eval()
            val_mae_sum = 0.0
            val_acc_sum = 0.0
            val_reg_loss_sum = 0.0
            val_class_loss_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for x, y_reg, y_class in val_loader:
                    x = x.to(DEVICE, non_blocking=True)
                    y_reg = y_reg.to(DEVICE, non_blocking=True)
                    y_class = y_class.to(DEVICE, non_blocking=True)

                    reg_out, class_out = model(x)
                    loss_reg = criterion_reg(reg_out.squeeze(), y_reg)
                    loss_class = criterion_class(class_out, y_class)

                    batch_size = y_reg.size(0)
                    val_mae_sum += torch.abs(reg_out.squeeze() - y_reg).sum().item()
                    preds = class_out.argmax(dim=1)
                    val_acc_sum += (preds == y_class).sum().item()
                    val_reg_loss_sum += loss_reg.item() * batch_size
                    val_class_loss_sum += loss_class.item() * batch_size
                    n_val += batch_size

            val_mae = val_mae_sum / n_val
            val_maes.append(val_mae)
            val_class_accs.append(val_acc_sum / n_val)
            val_reg_losses.append(val_reg_loss_sum / n_val)
            val_class_losses.append(val_class_loss_sum / n_val)

            scheduler.step(val_mae)
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | "
                f"Train MAE: {train_maes[-1]:.4f} | Val MAE: {val_mae:.4f} | "
                f"Train Acc: {train_class_accs[-1]:.4f} | Val Acc: {val_class_accs[-1]:.4f}"
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), best_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving plots with progress so far...")
        plot_and_save(
            train_maes, val_maes,
            train_class_accs, val_class_accs,
            train_reg_losses, val_reg_losses,
            train_class_losses, val_class_losses,
            out_path='summary_v3.png'
        )
        print("Saved summary_v3.png")
        return

    plot_and_save(
        train_maes, val_maes,
        train_class_accs, val_class_accs,
        train_reg_losses, val_reg_losses,
        train_class_losses, val_class_losses,
        out_path='summary_v3.png'
    )
    print(f"Saved summary_v3.png and best model to {best_path} (best val MAE={best_val_mae:.4f})")


if __name__ == '__main__':
    mp.freeze_support()
    main()
