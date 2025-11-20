import os
import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Button

from model_v1.dataloader import LettuceDataset

def test_lettuce_dataset():
    # Adjust paths as needed for your test environment
    dataset = LettuceDataset(
        RGB_dir="../../../datasets/Training/RGBImages",
        depth_dir="../../../datasets/Training/DepthImages",
        labels_file="../../../datasets/Training/Train.csv",
        image_size=64
    )
    assert len(dataset) > 0, "Dataset is empty!"
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor), "Image is not a torch.Tensor!"
    assert image.shape[0] == 4, f"Expected 4 channels (RGB+Depth), got {image.shape[0]}"
    assert image.shape[1] == image.shape[2] == 64, "Image not resized correctly!"
    assert isinstance(label, torch.Tensor), "Label is not a torch.Tensor!"
    print("Test passed: LettuceDataset loads and returns correct shapes.")
    return image, label

def visualize_sample(image, label):
    # image: torch.Tensor shape (4, H, W)
    rgb = image[:3].permute(1, 2, 0).numpy()  # (H, W, 3)
    depth = image[3].numpy()  # (H, W)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(rgb)
    axs[0].set_title("RGB")
    axs[0].axis('off')
    axs[1].imshow(depth, cmap='gray')
    axs[1].set_title("Depth")
    axs[1].axis('off')
    plt.suptitle(f"Label (DryWeightShoot): {label.item():.2f}")
    plt.tight_layout()
    plt.show()

def visualize_dataset_slideshow(dataset):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    idx = [0]  # mutable index
    total = len(dataset)

    def show(idx_val):
        image, label = dataset[idx_val]
        rgb = image[:3].permute(1, 2, 0).numpy()
        depth = image[3].numpy()
        axs[0].imshow(rgb)
        axs[0].set_title(f"RGB (Sample {idx_val+1}/{total})")
        axs[0].axis('off')
        axs[1].imshow(depth, cmap='gray')
        axs[1].set_title("Depth")
        axs[1].axis('off')
        fig.suptitle(f"Label (DryWeightShoot): {label.item():.2f}")
        plt.draw()

    def next_btn(event):
        idx[0] = (idx[0] + 1) % total
        show(idx[0])

    def prev_btn(event):
        idx[0] = (idx[0] - 1) % total
        show(idx[0])

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(bottom=0.2)
    axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Previous')
    bnext.on_clicked(next_btn)
    bprev.on_clicked(prev_btn)
    show(idx[0])
    plt.show()

if __name__ == "__main__":
    dataset = LettuceDataset(
        RGB_dir="../../../datasets/Training/RGBImages",
        depth_dir="../../../datasets/Training/DepthImages",
        labels_file="../../../datasets/Training/Train.csv",
        image_size=64
    )
    image, label = dataset[0]
    test_lettuce_dataset()
    visualize_dataset_slideshow(dataset)
