from RL.Env import Env
from RL.Agent import Agent
from Training import data_spilt, ReadH5d, create_data_loader
from Training import train_seg_net, test_seg_net
from Training import test_agent
from monai.transforms import *
from monai.networks.nets import DynUNet, SwinUNETR
from monai.losses import DiceFocalLoss
import torch
import random


batch_size=6
num_epoch=1000



seg_list, rl_list, holdout_list = data_spilt('./picai_h5', 110, 100, 10)
seg_list_promise, rl_list_promise, holdout_list_promise = data_spilt('./promise_h5', 231, 180, 20)

train_list = rl_list + rl_list_promise
test_list = holdout_list + holdout_list_promise



learn_length = 30
batch_size = 10
n_epochs = 1


device = 'cpu'
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
seg_model.load_state_dict(torch.load("./models/segmentation.ptm", map_location=device, weights_only=True))
seg_model.eval()

agent = Agent(gamma = 0.98, alpha=0.0001, batch_size=batch_size, n_epochs=n_epochs, device=device)
agent.load_models('trained_model/actor.ptm', device=device)



test_dir = './picai_h5/3.h5'
test_dir = random.choice(holdout_list)
# env = Env(test_dir, seg_model, learn_length)


test_dice = torch.tensor(test_agent(Env(test_dir, seg_model, learn_length), agent, learn_length, device=device))
# test_dice_random = torch.tensor(test_agent(Env(test_dir, seg_model), agent, 10, device=device, random=True))
print(test_dice)
# print(test_dice_random)

