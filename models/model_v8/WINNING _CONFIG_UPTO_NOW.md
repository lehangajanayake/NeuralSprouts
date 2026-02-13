train_csv: str = '../../datasets/Training/Augmented_v8/Train_aug.csv'
    rgb_dir: str = '../../datasets/Training/Augmented_v8/RGBImages'
    depth_dir: str = '../../datasets/Training/Augmented_v8/DepthImages'

    batch_size: int = 256
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    scheduler_min_lr: float = 1e-6

    val_ratio: float = 0.2
    seed: int = 43
    patience: int = 100
    outputs_per_original: int = 41
    num_folds: int = 1
    group_by_original: bool = True

    preload_to_gpu: bool = False
    preload_device: str = 'cuda'

    out_dir: str = '.'
    blacklist_ids: Tuple[int, ...] = (163,)
    best_mae_window: int = 5
    ema_decay: float = 0.995
    drop_path_prob: float = 0.1
    rgb_widths: Tuple[int, ...] = (32, 64, 96, 128)
    rgbd_widths: Tuple[int, ...] = (32, 64, 96, 128)
    embed_dim: int = 256
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5
    initial_frozen_rgb_blocks: int = 3
    initial_frozen_rgbd_blocks: int = 3
    unfreeze_interval: int = 5
    rgb_unfreeze_interval: int = 5
    rgbd_unfreeze_interval: int = 7
    unfreeze_start_epoch: int =7
    branch_warmup_epochs: int = 2
    branch_warmup_scale: float = 0.3
    huber_delta: float = 0.3