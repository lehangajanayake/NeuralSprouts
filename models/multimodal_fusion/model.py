"""
Multimodal Fusion Model Architecture.

RGB + Depth encoders with mid-level fusion, segmentation head, regression head,
and phenotype feature extraction for improved predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class SegmentationDecoder(nn.Module):
    """UNet-style decoder for segmentation."""
    
    def __init__(self, encoder_channels, decoder_channels=[256, 128, 64, 32]):
        super().__init__()
        
        # Adjust encoder channels to match expected input
        # encoder_channels are in reverse order (deepest to shallowest)
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        
        # Build decoder blocks
        self.blocks = nn.ModuleList()
        in_ch = encoder_channels[0]  # Start with deepest features
        
        for out_ch in decoder_channels:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )
            in_ch = out_ch
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[-1], 1, 1)
    
    def forward(self, features):
        """
        Args:
            features: Deepest fused features from encoder
        
        Returns:
            mask_logits: (B, 1, H, W) segmentation logits
        """
        x = features
        for block in self.blocks:
            x = block(x)
        
        mask_logits = self.seg_head(x)
        return mask_logits


class PhenotypeFeatureExtractor(nn.Module):
    """
    Extract phenotype features from predicted mask and depth image.
    Features: area fraction, bbox dimensions, depth statistics.
    """
    
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, mask_logits, depth):
        """
        Args:
            mask_logits: (B, 1, H, W) predicted mask logits
            depth: (B, 1, H, W) normalized depth image
        
        Returns:
            features: (B, F) phenotype features
        """
        B = mask_logits.shape[0]
        device = mask_logits.device
        
        # Get binary mask using sigmoid + threshold
        mask_probs = torch.sigmoid(mask_logits)
        # Use straight-through estimator for differentiability
        mask_binary = (mask_probs > self.threshold).float()
        # Detach for feature extraction but keep gradient flow through mask_probs
        mask_binary = mask_binary.detach() + mask_probs - mask_probs.detach()
        
        mask_binary = mask_binary.squeeze(1)  # (B, H, W)
        depth = depth.squeeze(1)  # (B, H, W)
        
        features_list = []
        
        for i in range(B):
            mask = mask_binary[i]
            d = depth[i]
            
            # Feature 1: Area fraction
            area_fraction = mask.mean()
            
            # Feature 2-3: Bounding box dimensions (normalized)
            mask_np = mask.cpu().numpy()
            rows = mask.sum(dim=1)
            cols = mask.sum(dim=0)
            
            # Handle empty mask
            if mask.sum() < 1:
                bbox_height = torch.tensor(0.0, device=device)
                bbox_width = torch.tensor(0.0, device=device)
            else:
                row_indices = torch.where(rows > 0)[0]
                col_indices = torch.where(cols > 0)[0]
                
                if len(row_indices) > 0 and len(col_indices) > 0:
                    bbox_height = (row_indices[-1] - row_indices[0] + 1).float() / mask.shape[0]
                    bbox_width = (col_indices[-1] - col_indices[0] + 1).float() / mask.shape[1]
                else:
                    bbox_height = torch.tensor(0.0, device=device)
                    bbox_width = torch.tensor(0.0, device=device)
            
            # Feature 4: Equivalent diameter from area
            equiv_diameter = 2 * torch.sqrt(area_fraction / 3.14159)
            
            # Features 5-7: Depth statistics inside mask
            masked_depth = d * mask
            valid_pixels = mask.sum()
            
            if valid_pixels > 0:
                depth_mean = masked_depth.sum() / (valid_pixels + 1e-8)
                depth_variance = ((masked_depth - depth_mean * mask) ** 2).sum() / (valid_pixels + 1e-8)
                depth_std = torch.sqrt(depth_variance + 1e-8)
                
                # Approximate median using quantile
                masked_depth_flat = masked_depth[mask > 0.5]
                if len(masked_depth_flat) > 0:
                    depth_median = torch.median(masked_depth_flat)
                else:
                    depth_median = torch.tensor(0.0, device=device)
            else:
                depth_mean = torch.tensor(0.0, device=device)
                depth_std = torch.tensor(0.0, device=device)
                depth_median = torch.tensor(0.0, device=device)
            
            # Combine features
            feats = torch.stack([
                area_fraction,
                bbox_height,
                bbox_width,
                equiv_diameter,
                depth_mean,
                depth_std,
                depth_median
            ])
            
            features_list.append(feats)
        
        features = torch.stack(features_list, dim=0)  # (B, 7)
        return features


class RegressionHead(nn.Module):
    """MLP for regression from deep features."""
    
    def __init__(self, in_features, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Final prediction layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (B, F) features
        
        Returns:
            predictions: (B,) regression output
        """
        return self.mlp(x).squeeze(-1)


class PhenotypeHead(nn.Module):
    """MLP for regression from phenotype features."""
    
    def __init__(self, in_features=7, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (B, 7) phenotype features
        
        Returns:
            predictions: (B,) regression output
        """
        return self.mlp(x).squeeze(-1)


class MultimodalFusionModel(nn.Module):
    """
    Complete multimodal fusion model with:
    - Dual encoders (RGB + Depth)
    - Mid-level fusion
    - Segmentation decoder
    - Deep regression head
    - Phenotype feature extraction and regression
    - Learnable blending
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # RGB Encoder
        self.rgb_encoder = timm.create_model(
            config.RGB_BACKBONE,
            pretrained=config.PRETRAINED,
            features_only=True
            # Use default out_indices for ConvNeXt
        )
        
        # Depth Encoder (modify first conv for 1-channel input)
        self.depth_encoder = timm.create_model(
            config.DEPTH_BACKBONE,
            pretrained=False,  # Can't use pretrained for 1-channel
            features_only=True,
            in_chans=1
            # Use default out_indices for ConvNeXt
        )
        
        # Get feature dimensions
        dummy_rgb = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        dummy_depth = torch.randn(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
        
        with torch.no_grad():
            rgb_feats = self.rgb_encoder(dummy_rgb)
            depth_feats = self.depth_encoder(dummy_depth)
        
        # Feature dimensions at each scale
        self.rgb_channels = [f.shape[1] for f in rgb_feats]
        self.depth_channels = [f.shape[1] for f in depth_feats]
        
        # Fusion layers for each scale
        self.fusion_layers = nn.ModuleList()
        for rgb_ch, depth_ch in zip(self.rgb_channels, self.depth_channels):
            fusion = nn.Sequential(
                nn.Conv2d(rgb_ch + depth_ch, config.FUSION_CHANNELS, 1),
                nn.BatchNorm2d(config.FUSION_CHANNELS),
                nn.ReLU(inplace=True)
            )
            self.fusion_layers.append(fusion)
        
        # Segmentation decoder (uses deepest fused features)
        if config.USE_PHENOTYPE_FEATURES:
            self.seg_decoder = SegmentationDecoder(
                encoder_channels=[config.FUSION_CHANNELS],
                decoder_channels=config.DECODER_CHANNELS
            )
            self.phenotype_extractor = PhenotypeFeatureExtractor(
                threshold=config.PHENOTYPE_THRESHOLD
            )
            self.phenotype_head = PhenotypeHead(
                in_features=7,
                hidden_dims=config.PHENOTYPE_HIDDEN
            )
        
        # Deep regression head (uses global pooled deepest features)
        self.regression_head = RegressionHead(
            in_features=config.FUSION_CHANNELS,
            hidden_dims=config.REGRESSION_HIDDEN
        )
        
        # Learnable blending weight
        if config.LEARNABLE_ALPHA:
            self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('alpha', torch.tensor(config.FIXED_ALPHA))
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: (B, 3, H, W) RGB images
            depth: (B, 1, H, W) Depth images
        
        Returns:
            Dictionary with predictions and intermediate outputs
        """
        # Extract multi-scale features
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)
        
        # Fuse features at each scale
        fused_features = []
        for i, (rgb_feat, depth_feat, fusion_layer) in enumerate(
            zip(rgb_features, depth_features, self.fusion_layers)
        ):
            # Concatenate and fuse
            concat_feat = torch.cat([rgb_feat, depth_feat], dim=1)
            fused = fusion_layer(concat_feat)
            fused_features.append(fused)
        
        # Use deepest fused features for both tasks
        deep_features = fused_features[-1]  # (B, C, H/32, W/32)
        
        outputs = {}
        
        # Segmentation branch
        if self.config.USE_PHENOTYPE_FEATURES:
            mask_logits = self.seg_decoder(deep_features)
            outputs['mask_logits'] = mask_logits
            
            # Extract phenotype features
            phenotype_features = self.phenotype_extractor(mask_logits, depth)
            outputs['phenotype_features'] = phenotype_features
            
            # Phenotype regression
            phen_pred = self.phenotype_head(phenotype_features)
            outputs['phen_pred'] = phen_pred
        
        # Deep regression branch
        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(deep_features, 1).flatten(1)
        deep_pred = self.regression_head(pooled_features)
        outputs['deep_pred'] = deep_pred
        
        # Final blended prediction
        if self.config.USE_PHENOTYPE_FEATURES and 'phen_pred' in outputs:
            if self.config.LEARNABLE_ALPHA:
                alpha = torch.sigmoid(self.alpha_logit)
            else:
                alpha = self.alpha
            
            final_pred = alpha * deep_pred + (1 - alpha) * phen_pred
            outputs['alpha'] = alpha
        else:
            final_pred = deep_pred
        
        outputs['final_pred'] = final_pred
        
        return outputs


def build_model(config):
    """Build and return the model."""
    model = MultimodalFusionModel(config)
    return model
