import torch
import torch.nn as nn

class FinalModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(FinalModel, self).__init__()

        # Helper to create a clean Conv -> Batch Norm -> ReLU block.
        # Calling this function creates NEW layers every time, ensuring
        # NO weights are shared between branches.
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # Safety net

        # --- RGBD BRANCH (4 Channels) ---
        # Input: 4x64x64 -> Output: 256x1x1
        self.rgbd_conv = nn.Sequential(
            conv_block(4, 16),    self.pool,  # 64 -> 32
            conv_block(16, 32),   self.pool,  # 32 -> 16
            conv_block(32, 64),   self.pool,  # 16 -> 8
            conv_block(64, 128),  self.pool,  # 8 -> 4
            conv_block(128, 256), self.pool,  # 4 -> 2
            conv_block(256, 256), self.pool   # 2 -> 1
        )

        # --- RGB BRANCH (3 Channels) ---
        # Input: 3x64x64 -> Output: 256x1x1
        # Added the 6th layer here so it matches the depth of RGBD
        self.rgb_conv = nn.Sequential(
            conv_block(3, 16),    self.pool,
            conv_block(16, 32),   self.pool,
            conv_block(32, 64),   self.pool,
            conv_block(64, 128),  self.pool,
            conv_block(128, 256), self.pool,
            conv_block(256, 256), self.pool
        )

        # --- FULLY CONNECTED LAYERS ---

        # 1. RGBD Regression Branch
        self.fc_rgbd = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fcWeightEnd = nn.Linear(64, 1)

        # 2. RGB Classification Branch
        self.fc_rgb = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fcTypeEnd = nn.Linear(64, num_classes)

        # 3. Fusion Regression Branch
        self.fusion_fc = nn.Sequential(
            nn.Linear(64 + 64, 256), # Concatenating the 64-dim features
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # --- RGBD Path ---
        rgbd = self.rgbd_conv(x)                 # Output: [B, 256, 1, 1]
        rgbd = self.adaptive_pool(rgbd)          # Ensure 1x1
        rgbd_flat = torch.flatten(rgbd, 1)       # Output: [B, 256]

        rgbd_feat = self.fc_rgbd(rgbd_flat)      # Output: [B, 64]
        regression_out = self.fcWeightEnd(rgbd_feat)

        # --- RGB Path ---
        rgb = self.rgb_conv(x[:, :3, :, :])      # Output: [B, 256, 1, 1]
        rgb = self.adaptive_pool(rgb)            # Ensure 1x1
        rgb_flat = torch.flatten(rgb, 1)         # Output: [B, 256]

        rgb_feat = self.fc_rgb(rgb_flat)         # Output: [B, 64]
        class_out = self.fcTypeEnd(rgb_feat)

        # --- Fusion Path ---
        # Concatenate the features just before the final heads
        fusion_input = torch.cat((rgbd_feat, rgb_feat), dim=1) # [B, 128]
        fusion_out = self.fusion_fc(fusion_input)

        return regression_out, class_out, fusion_out