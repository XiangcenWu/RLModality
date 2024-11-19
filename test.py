from Training import data_spilt, ReadH5d, create_data_loader
from Training import train_weak_net, test_weak_net, test_single_weak_net, test_single_weak_net_with_continouw_index
from Training.NetWorks import WeakModel
from monai.transforms import *
from monai.networks.nets import DynUNet, SwinUNETR
from monai.losses import DiceFocalLoss
import torch

num_epoch=1000
device = 'cuda:0'


seg_list, rl_list, holdout_list = data_spilt('/home/xiangcen/RLModality/picai_h5', 190, 0, 20)
inference_transform = ReadH5d()

iii = 5
holdout_list = [holdout_list[iii]]
print(holdout_list[0])
inference_loader = create_data_loader(holdout_list, inference_transform, batch_size=1, shuffle=False)

seg_model = SwinUNETR(
    img_size = (128, 128, 32),
    in_channels = 2,
    out_channels = 1,
    depths = (2, 2, 2, 2),
    num_heads = (3, 6, 12, 24),
    drop_rate = 0.1,
    attn_drop_rate = 0.1,
    dropout_path_rate = 0.1,
    downsample="mergingv2",
    use_v2=True,
)
seg_model.load_state_dict(torch.load("/home/xiangcen/RLModality/models/segmentation.ptm", map_location=device, weights_only=True))
weak_model = WeakModel()
weak_model.load_state_dict(torch.load("/home/xiangcen/RLModality/models/weak.ptm", map_location=device, weights_only=True))



test_acc = test_single_weak_net_with_continouw_index(weak_model, seg_model, inference_loader, device=device)



print(f'acc: {test_acc}')
    

