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



step_per_patient = 60
batch_size = 10
n_epochs = 5




agent = Agent(gamma = 0., batch_size=batch_size, n_epochs=n_epochs, device=device)
actor_optimizer = torch.optim.AdamW(agent.actor.parameters(), lr=0.0001)
test_result_list = []

train_list = rl_list
# train_list = ['./picai_h5/3.h5']



for epoch in range(99999999): # loop over dataset (patients)
    for train_dir in train_list:
        env = Env(train_dir, seg_model)
        if env.all_zero:
            continue
        obs = env.reset()
        for _ in range(step_per_patient):
            action, features= agent.choose_action(obs.unsqueeze(0).to(device), noise=0.3)
            
            # if torch.rand(size=(1, )).item() > 0.5:
            #     action = 0
            next_obs, reward = env.step_train(action)
            agent.remember(obs, action, reward)
            obs = next_obs
        agent.learn_reinforce(actor_optimizer)
        
        #     # test_dice = torch.zeros(size=(10, ))

        #     # for test_dir in test_list:

        #     #     test_dice += torch.tensor(test_agent(Env(test_dir, seg_model, eps_length), agent, 10, device=device))
        #     #     # test_dice_random += torch.tensor(test_agent(Env(train_dir, seg_model, eps_length), None, 10, device=device, random=True))
        #     # current_test_tensor = test_dice/len(test_list)
        #     # print(f'Trained on {num_env}, test list {current_test_tensor.tolist()}')
        #     # test_result_list.append(current_test_tensor)
            
        #     # torch.save(torch.stack(test_result_list), '/home/xiangcen/RLModality/models/loss/agent_test.pt')
    
    
    if (epoch + 1) % 50 == 0:
        agent.save_models(
            f'/home/xiangcen/RLModality/models/rl_models/actor{epoch}.ptm'
        )








