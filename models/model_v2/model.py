import torch
import torch.nn as nn

class ModelV2(nn.Module):
    def __init__(self, num_classes=3):
        super(ModelV2, self).__init__()
        # RGBD branch (main regression)
        self.convRGBD = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc_rgbd1 = nn.Linear(256 * 1 * 1, 256)
        self.fc_rgbd2 = nn.Linear(256, 128)
        self.fc_rgbd3 = nn.Linear(128, 64)
        self.fcWeightEnd = nn.Linear(64, 1)

        # RGB branch (aux classification)
        self.convRGB = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc_rgb1 = nn.Linear(256 * 2 * 2, 256)
        self.fc_rgb2 = nn.Linear(256, 128)
        self.fc_rgb3 = nn.Linear(128, 64)
        self.fcTypeEnd = nn.Linear(64, num_classes)

        # Depth branch (aux leaf area regression)
        self.convDepth = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.depth_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.depth_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.depth_conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.depth_conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.depth_conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc_depth1 = nn.Linear(256 * 1 * 1, 128)
        self.fc_depth2 = nn.Linear(128, 64)
        self.fcLeafAreaEnd = nn.Linear(64, 1)

        self.relu = nn.ReLU()

        self.rgbd_conv = nn.Sequential(
            self.convRGBD, self.pool, self.relu,
            self.conv2, self.pool, self.relu,
            self.conv3, self.pool, self.relu,
            self.conv4, self.pool, self.relu,
            self.conv5, self.pool, self.relu,
            self.conv6, self.pool, self.relu,
        )
        self.rgb_conv = nn.Sequential(
            self.convRGB, self.pool, self.relu,
            self.conv2, self.pool, self.relu,
            self.conv3, self.pool, self.relu,
            self.conv4, self.pool, self.relu,
            self.conv5, self.pool, self.relu,
        )
        self.depth_branch = nn.Sequential(
            self.convDepth, self.pool, self.relu,
            self.depth_conv2, self.pool, self.relu,
            self.depth_conv3, self.pool, self.relu,
            self.depth_conv4, self.pool, self.relu,
            self.depth_conv5, self.pool, self.relu,
            self.depth_conv6, self.pool, self.relu,
        )
        self.fusion_fc1 = nn.Linear(64 + 64, 256)
        self.fusion_fc2 = nn.Linear(256, 128)
        self.fusion_fc3 = nn.Linear(128, 64)
        self.fusion_fc4 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, 4, H, W] (RGBD)
        rgbd = self.rgbd_conv(x)
        rgbd_flat = rgbd.view(rgbd.size(0), -1)
        rgbd_feat = self.relu(self.fc_rgbd1(rgbd_flat))
        rgbd_feat2 = self.relu(self.fc_rgbd2(rgbd_feat))
        rgbd_feat3 = self.relu(self.fc_rgbd3(rgbd_feat2))
        regression_out = self.fcWeightEnd(rgbd_feat3)

        # RGB branch (aux classification)
        rgb = self.rgb_conv(x[:, :3, :, :])
        rgb_flat = rgb.view(rgb.size(0), -1)
        rgb_feat = self.relu(self.fc_rgb1(rgb_flat))
        rgb_feat2 = self.relu(self.fc_rgb2(rgb_feat))
        rgb_feat3 = self.relu(self.fc_rgb3(rgb_feat2))
        class_out = self.fcTypeEnd(rgb_feat3)

        # Depth branch (aux leaf area regression)
        depth = self.depth_branch(x[:, 3:, :, :])
        depth_flat = depth.view(depth.size(0), -1)
        depth_feat = self.relu(self.fc_depth1(depth_flat))
        depth_feat2 = self.relu(self.fc_depth2(depth_feat))
        leaf_area_out = self.fcLeafAreaEnd(depth_feat2)

        # Fusion regression (use last FC features from both main branches)
        fusion_input = torch.cat((rgbd_feat3, rgb_feat3), dim=1)
        fusion = self.relu(self.fusion_fc1(fusion_input))
        fusion = self.relu(self.fusion_fc2(fusion))
        fusion = self.relu(self.fusion_fc3(fusion))
        fusion_out = self.fusion_fc4(fusion)

        return regression_out, class_out, leaf_area_out, fusion_out
