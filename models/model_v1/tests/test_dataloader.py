import os
import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Button

from model_v1.dataloader import LettuceDataset, TestLettuceDataset

def test_lettuce_dataset():
    # Adjust paths as needed for your test environment
    dataset = LettuceDataset(
        RGB_dir="../datasets/Training/RGBImages",
        depth_dir="../datasets/Training/DepthImages",
        labels_file="../datasets/Training/Train.csv",
        image_size=64
    )
    assert len(dataset) > 0, "Dataset is empty!"
    image, dry_weight, variety_class, image_id = dataset[0]
    assert isinstance(image, torch.Tensor), "Image is not a torch.Tensor!"
    assert image.shape[0] == 4, f"Expected 4 channels (RGB+Depth), got {image.shape[0]}"
    assert image.shape[1] == image.shape[2] == 64, "Image not resized correctly!"
    assert isinstance(dry_weight, torch.Tensor), "Dry weight label is not a torch.Tensor!"
    assert isinstance(variety_class, torch.Tensor), "Variety class label is not a torch.Tensor!"
    print("Test passed: LettuceDataset loads and returns correct shapes.")
    return image, dry_weight, variety_class, image_id

def test_lettuce_test_dataset():
    dataset = TestLettuceDataset(
        RGB_dir="../datasets/Test/RGBImages",
        depth_dir="../datasets/Test/DepthImages",
        image_size=64
    )
    assert len(dataset) > 0, "Test Dataset is empty!"
    image, image_id = dataset[0]
    assert isinstance(image, torch.Tensor), "Image is not a torch.Tensor!"
    assert image.shape[0] == 4, f"Expected 4 channels (RGB+Depth), got {image.shape[0]}"
    assert image.shape[1] == image.shape[2] == 64, "Image not resized correctly!"
    print("Test passed: Test LettuceDataset loads and returns correct shapes.")
    return image, image_id

def visualize_sample(image, dry_weight, variety_class):
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
    plt.suptitle(f"Label (DryWeightShoot, type): ({dry_weight.item():.2f}, {variety_class.item()})")
    plt.tight_layout()
    plt.show()

def visualize_dataset_slideshow(dataset):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    idx = [0]  # mutable index
    total = len(dataset)

    def get_item(i):
        item = dataset[i]
        # normalize to (image, dry_weight, variety_or_id)
        if isinstance(item, (tuple, list)):
            if len(item) == 2:
                return item[0], None, item[1]
            elif len(item) >= 3:
                return item[0], item[1], item[2]
            else:
                return item[0], None, None
        else:
            return item, None, None

    def has_no_dryweight(dw):
        if dw is None:
            return True
        if isinstance(dw, torch.Tensor):
            if dw.numel() == 0:
                return True
            try:
                return torch.isnan(dw).all().item()
            except Exception:
                return False
        return False

    def find_next_valid(start, step):
        i = start
        for _ in range(total):
            _, dry_weight, _ = get_item(i)
            if has_no_dryweight(dry_weight):
                return i
            i = (i + step) % total
        return start  # no valid found, return original

    def show(idx_val):
        image, dry_weight, variety_or_id = get_item(idx_val)
        if not has_no_dryweight(dry_weight):
            # indicate skipped sample
            axs[0].cla()
            axs[1].cla()
            axs[0].text(0.5, 0.5, f"Sample {idx_val+1}/{total} skipped\n(dry weight present)",
                        ha='center', va='center', fontsize=10)
            axs[0].axis('off')
            axs[1].axis('off')
            fig.suptitle("")
            plt.draw()
            return

        rgb = image[:3].permute(1, 2, 0).numpy()
        depth = image[3].numpy() if image.shape[0] > 3 else None
        axs[0].cla()
        axs[1].cla()
        axs[0].imshow(rgb)
        axs[0].set_title(f"RGB (Sample {idx_val+1}/{total})")
        axs[0].axis('off')
        if depth is not None:
            axs[1].imshow(depth, cmap='gray')
            axs[1].set_title("Depth")
        else:
            axs[1].text(0.5, 0.5, "No depth channel", ha='center', va='center')
        axs[1].axis('off')
        label_text = ""
        if dry_weight is not None:
            try:
                label_text = f"Label (DryWeightShoot, type): ({dry_weight.item():.2f}, {variety_or_id})"
            except Exception:
                label_text = f"Label present (type: {type(dry_weight)})"
        else:
            label_text = f"Sample ID: {variety_or_id}" if variety_or_id is not None else ""
        fig.suptitle(label_text)
        plt.draw()

    def next_btn(event):
        start = (idx[0] + 1) % total
        idx[0] = find_next_valid(start, 1)
        show(idx[0])

    def prev_btn(event):
        start = (idx[0] - 1) % total
        idx[0] = find_next_valid(start, -1)
        show(idx[0])

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(bottom=0.2)
    axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Previous')
    bnext.on_clicked(next_btn)
    bprev.on_clicked(prev_btn)
    # start at first valid sample (or 0 if none)
    idx[0] = find_next_valid(idx[0], 1)
    show(idx[0])
    plt.show()

if __name__ == "__main__":
    dataset = TestLettuceDataset(
        RGB_dir="../datasets/Training/RGBImages",
        depth_dir="../datasets/Training/DepthImages",
        image_size=64
    )
    image, image_id = dataset[0]
    #test_lettuce_dataset()

    test_lettuce_test_dataset()
    visualize_dataset_slideshow(dataset)
