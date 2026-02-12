"""CLI entry point for the offline preprocessing pipeline."""

from config import Config
from preprocess import PreprocessConfig, run_preprocessing


def main():
    app_cfg = Config()
    app_cfg.print_config()

    print("\n" + "=" * 60)
    print("Starting Preprocessing Pipeline")
    print("=" * 60)

    preprocess_cfg = PreprocessConfig(
        train_rgb_dir=app_cfg.TRAIN_RGB_DIR,
        train_depth_dir=app_cfg.TRAIN_DEPTH_DIR,
        labels_csv=app_cfg.TRAIN_CSV,
        out_rgb_dir=app_cfg.AUGMENTED_RGB_DIR,
        out_depth_dir=app_cfg.AUGMENTED_DEPTH_DIR,
        out_csv=app_cfg.AUGMENTED_CSV,
        crop_size=app_cfg.CENTER_CROP_SIZE,
        image_size=app_cfg.RESIZE_SIZE,
        num_aug_per_image=app_cfg.PREPROCESS_NUM_AUG,
        seed=app_cfg.PREPROCESS_SEED,
        num_workers=app_cfg.PREPROCESS_NUM_WORKERS,
        max_items=app_cfg.PREPROCESS_MAX_ITEMS,
    )

    run_preprocessing(preprocess_cfg)

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"RGB output: {app_cfg.AUGMENTED_RGB_DIR}")
    print(f"Depth output: {app_cfg.AUGMENTED_DEPTH_DIR}")
    print(f"Augmented CSV: {app_cfg.AUGMENTED_CSV}")
    print("\nNext: run training (python main.py train)")


if __name__ == "__main__":
    main()
