import torch.nn as nn
import torch
from monai.networks.nets.swin_unetr import SwinTransformer




class vit_classifier(nn.Module):
    
    
    def __init__(self, num_output=32):
        super().__init__()
        
        
        


        self.vit = SwinTransformer(
            3, 24, (7, 7, 7), (2, 2, 2), (2, 2, 2, 2), (3, 6, 12, 24), downsample="mergingv2", use_v2=True,
        )
        
        self.conv0 = nn.Conv3d(24, 48,kernel_size=(3, 3, 3),  # 3x3x3 kernel
                                stride=(2, 2, 2),  # Stride 2 for downsampling
                                padding=1)
        self.conv1 = nn.Conv3d(48, 96,kernel_size=(3, 3, 3),  # 3x3x3 kernel
                                stride=(2, 2, 2),  # Stride 2 for downsampling
                                padding=1)
        self.conv2 = nn.Conv3d(96, 192,kernel_size=(3, 3, 3),  # 3x3x3 kernel
                                stride=(2, 2, 2),  # Stride 2 for downsampling
                                padding=1)
        self.conv3 = nn.Conv3d(192, 384,kernel_size=(3, 3, 3),  # 3x3x3 kernel
                                stride=(2, 2, 2),  # Stride 2 for downsampling
                                padding=1)
        self.conv4 = nn.Conv3d(384, 384,kernel_size=(4, 4, 1),  # 3x3x3 kernel
                                stride=1,  # Stride 2 for downsampling
                                padding=0)
        
        self.output = nn.Linear(384, num_output)
        
        
        
        
        
    def forward(self, x):
        feature0, feature1, feature2, feature3, feature4 = \
            self.vit(x)
        feature1 = nn.functional.relu(self.conv0(feature0) + feature1)
        
        feature2 = nn.functional.relu(self.conv1(feature1) + feature2)
        
        feature3 = nn.functional.relu(self.conv2(feature2) + feature3)
        
        feature4 = nn.functional.relu(self.conv3(feature3) + feature4)
        
        feature = nn.functional.relu(self.conv4(feature4))
        
        feature = feature.flatten(1)

        return  self.output(feature)


# torch.Size([1, 24, 64, 64, 16])
# torch.Size([1, 48, 32, 32, 8])
# torch.Size([1, 96, 16, 16, 4])
# torch.Size([1, 192, 8, 8, 2])
# torch.Size([1, 384, 4, 4, 1])
# if __name__ == "__main__":
#     import torch
#     model = vit_classifier()
    
#     x = torch.rand(2, 3, 128, 128, 32)
    
#     y = model(x)
    
#     print(y.shape)
    
    