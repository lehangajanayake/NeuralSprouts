import torch
import torch.nn as nn


class model_v1(nn.Module):
    def __init__(self, num_classes=3):
        super(model_v1, self).__init__()
        self.convRGBD = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.convRGB = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # FCs for RGBD branch (after 6x pooling, 256x1x1)
        self.fc_rgbd1 = nn.Linear(256 * 1 * 1, 256)
        self.fc_rgbd2 = nn.Linear(256, 128)
        self.fc_rgbd3 = nn.Linear(128, 64)
        self.fcWeightEnd = nn.Linear(64, 1)

    # FCs for RGB branch (after 5x pooling, 256x2x2)
        self.fc_rgb1 = nn.Linear(256 * 2 * 2, 256)
        self.fc_rgb2 = nn.Linear(256, 128)
        self.fc_rgb3 = nn.Linear(128, 64)
        self.fcTypeEnd = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

        # Main regression branch (RGBD)
        self.rgbd_conv = nn.Sequential(
            self.convRGBD,  # 4x64x64 -> 16x64x64
            self.pool,      # 16x32x32
            self.relu,
            self.conv2,     # 32x32x32
            self.pool,      # 32x16x16
            self.relu,
            self.conv3,     # 64x16x16
            self.pool,      # 64x8x8
            self.relu,
            self.conv4,     # 128x8x8
            self.pool,      # 128x4x4
            self.relu,
            self.conv5,     # 256x4x4
            self.pool,      # 256x2x2
            self.relu,
            self.conv6,     # 256x2x2
            self.pool,      # 256x1x1
            self.relu,
        )

        # Aux classification branch (RGB only)
        self.rgb_conv = nn.Sequential(
            self.convRGB,   # 3x64x64 -> 16x64x64
            self.pool,      # 16x32x32
            self.relu,
            self.conv2,     # 32x32x32
            self.pool,      # 32x16x16
            self.relu,
            self.conv3,     # 64x16x16
            self.pool,      # 64x8x8
            self.relu,
            self.conv4,     # 128x8x8
            self.pool,      # 128x4x4
            self.relu,
            self.conv5,     # 256x4x4
            self.pool,      # 256x2x2
            self.relu,
        )

        # Fusion branch (deeper)
        self.fusion_fc1 = nn.Linear(64 + 64, 256)
        self.fusion_fc2 = nn.Linear(256, 128)
        self.fusion_fc3 = nn.Linear(128, 64)
        self.fusion_fc4 = nn.Linear(64, 1)
    def forward(self, x):
        # Main regression (RGBD)
        rgbd = self.rgbd_conv(x)  # [B, 256, 1, 1]
        rgbd_flat = rgbd.view(rgbd.size(0), -1)
        rgbd_feat = self.relu(self.fc_rgbd1(rgbd_flat))
        rgbd_feat2 = self.relu(self.fc_rgbd2(rgbd_feat))
        rgbd_feat3 = self.relu(self.fc_rgbd3(rgbd_feat2))
        regression_out = self.fcWeightEnd(rgbd_feat3)

        # Aux classification (RGB only)
        rgb = self.rgb_conv(x[:, :3, :, :])  # [B, 256, 2, 2]
        rgb_flat = rgb.view(rgb.size(0), -1)
        rgb_feat = self.relu(self.fc_rgb1(rgb_flat))
        rgb_feat2 = self.relu(self.fc_rgb2(rgb_feat))
        rgb_feat3 = self.relu(self.fc_rgb3(rgb_feat2))
        class_out = self.fcTypeEnd(rgb_feat3)

        # Fusion regression (use last FC features from both branches)
        fusion_input = torch.cat((rgbd_feat3, rgb_feat3), dim=1)
        fusion = self.relu(self.fusion_fc1(fusion_input))
        fusion = self.relu(self.fusion_fc2(fusion))
        fusion = self.relu(self.fusion_fc3(fusion))
        fusion_out = self.fusion_fc4(fusion)

        return regression_out, class_out, fusion_out