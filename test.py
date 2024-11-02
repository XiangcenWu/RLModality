from monai.networks.nets.vit import ViT
import torch




model = ViT(
    in_channels=2,
    img_size=(128, 128, 64),
    patch_size=(32, 32, 16),
    classification=True, 
    num_classes=3,
    post_activation=None
)



x = torch.rand(1, 2, 128, 128, 64)


o = model(x)



print(o[0])


