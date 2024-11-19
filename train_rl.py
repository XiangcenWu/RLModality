from RL.Env import Env
import torch
from Training.NetWorks import WeakModel
from monai.networks.nets import SwinUNETR


device = 'cuda:0'
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
weak_model.eval()
seg_model.eval()

env = Env("/home/xiangcen/RLModality/picai_h5/39.h5", seg_model, weak_model)



obs = env.reset()

while True:

    action = (0, 0)
    next_obs, reward = env.step(action)

    print(reward)
    