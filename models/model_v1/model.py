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

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fcWeightEnd = nn.Linear(128, 1)
        self.fcTypeEnd = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        # Main regression branch (RGBD)
        self.rgbd_branch = nn.Sequential(
            self.convRGBD,
            self.pool,
            self.relu,
            self.conv2,
            self.pool,
            self.relu,
            self.conv3,
            self.pool,
            self.relu,
            nn.Flatten(),
            self.fc1,
            self.relu
        )
        self.regression_head = self.fcWeightEnd

        # Aux classification branch (RGB only)
        self.rgb_branch = nn.Sequential(
            self.convRGB,
            self.pool,
            self.relu,
            self.conv2,
            self.pool,
            self.relu,
            self.conv3,
            self.pool,
            self.relu,
            nn.Flatten(),
            self.fc1,
            self.relu
        )
        self.classification_head = self.fcTypeEnd

        # Fusion branch
        self.fusion_fc1 = nn.Linear(128 + 128, 128)
        self.fusion_fc2 = nn.Linear(128, 64)
        self.fusion_fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()



    def forward(self, x):
        # Main regression (RGBD)
        rgbd_feat = self.rgbd_branch(x)
        regression_out = self.regression_head(rgbd_feat)

        # Aux classification (RGB only)
        rgb_feat = self.rgb_branch(x[:, :3, :, :])
        class_out = self.classification_head(rgb_feat)

        # Fusion regression
        fusion_input = torch.cat((rgbd_feat, rgb_feat), dim=1)
        fusion = self.relu(self.fusion_fc1(fusion_input))
        fusion = self.relu(self.fusion_fc2(fusion))
        fusion_out = self.fusion_fc3(fusion)

        return regression_out, class_out, fusion_out