from RL.Env import Env
from Training import data_spilt, ReadH5d, create_data_loader
from Training import train_seg_net, test_seg_net
from monai.transforms import *
from monai.networks.nets import DynUNet, SwinUNETR
from monai.losses import DiceFocalLoss
import torch
batch_size=6
num_epoch=1000
device = 'cuda:0'


seg_list, rl_list, holdout_list = data_spilt('/home/xiangcen/RLModality/picai_h5', 110, 100, 10)
seg_list_promise, rl_list_promise, holdout_list_promise = data_spilt('/home/xiangcen/RLModality/promise_h5', 231, 180, 20)

_dir = rl_list[34]
# _dir = rl_list_promise[4]

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
seg_model.eval()

print(_dir)
env = Env(_dir, seg_model)



obs = env.reset()

for _ in range(300):

    action = (
        torch.randint(0, 3, size=(1, )).item(), 
        torch.randint(0, 3, size=(1, )).item(),
        torch.randint(0, 8, size=(1, )).item(),
        torch.randint(0, 4, size=(1, )).item(),
    )
    next_obs, reward = env.step_train(action)

    print(reward)
print(env.get_all_accuracy())
