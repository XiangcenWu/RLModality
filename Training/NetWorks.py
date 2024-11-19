import torch
import torch.nn as nn
from monai.networks.nets import ViT



class WeakModel(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        self.vit = ViT(
            in_channels=3,                    # For single-channel (e.g., grayscale medical images)
            img_size=(128, 128, 32),         # 3D input size (D, H, W)
            patch_size=(16, 16, 4),          # 3D patch size
            hidden_size=1024,                 # Large hidden size for the transformer
            mlp_dim=4096,                     # MLP dimension (typically 4x hidden_size)
            num_layers=24,                    # Number of transformer layers
            num_heads=16,                     # Number of attention heads
            proj_type="conv",                 # Use 3D convolution for patch projection
            pos_embed_type="learnable",       # Learnable positional embeddings
            classification=True,              # Enable classification mode
            num_classes=1,                 # Number of output classes
            dropout_rate=0.,                 # Dropout for regularization
            spatial_dims=3,                   # 3D data
            post_activation=None,           # Post-activation function
            qkv_bias=True,                    # Use bias in query, key, value
            save_attn=True,                  # Don't save attention maps by default
        )
    
    def forward(self, x):
        x = self.vit(x)[0]
        return torch.sigmoid(x)
        
    
    
    