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



seg_list, rl_list, holdout_list = data_spilt('/home/xiangcen/RLModality/picai_h5', 110, 100, 10)
seg_list_promise, rl_list_promise, holdout_list_promise = data_spilt('/home/xiangcen/RLModality/promise_h5', 231, 180, 20)

train_list = rl_list + rl_list_promise
test_list = holdout_list + holdout_list_promise



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


eps_length = 10
learn_length = 10
batch_size = 10
n_epochs = 5


agent = Agent(gamma = 0.88, alpha=0.0001, gae_lambda=0.85, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, device=device)
test_result_list = []

train_list = rl_list
test_list = holdout_list



print(len(train_list), len(test_list))
for num_env in range(99999999): # loop over dataset (patients)
    agent.memory.clear_memory()
    train_dir = random.choice(train_list)
    env = Env(train_dir, seg_model, eps_length)
    if env.all_zero:
        continue
    obs = env.reset()
    for _ in range(eps_length):
        action, prob, val = agent.choose_action(obs.unsqueeze(0).to(device), noise=0.5)
        next_obs, reward = env.step_train(action)
        agent.remember(obs, action, prob, val, reward)
        if (_ + 1) % learn_length == 0:
            agent.learn()
        obs = next_obs

    # if (num_env + 1) % 50 == 0:


    #     test_dice = torch.zeros(size=(10, ))

    #     for test_dir in test_list:

    #         test_dice += torch.tensor(test_agent(Env(test_dir, seg_model, eps_length), agent, 10, device=device))
    #         # test_dice_random += torch.tensor(test_agent(Env(train_dir, seg_model, eps_length), None, 10, device=device, random=True))
    #     current_test_tensor = test_dice/len(test_list)
    #     print(f'Trained on {num_env}, test list {current_test_tensor.tolist()}')
    #     test_result_list.append(current_test_tensor)
        
    #     torch.save(torch.stack(test_result_list), '/home/xiangcen/RLModality/models/loss/agent_test.pt')
    #     agent.save_models(
    #         '/home/xiangcen/RLModality/models/rl_models/actor.ptm',
    #         '/home/xiangcen/RLModality/models/rl_models/critic.ptm'
    #     )








