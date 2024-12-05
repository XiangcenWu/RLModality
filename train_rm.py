from Training import data_spilt, ReadH5d, create_data_loader
from Training import train_reward_model, test_reward_model
from Training.NetWorks import RewardModel
from monai.transforms import *
from monai.networks.nets import DynUNet, SwinUNETR
from monai.losses import DiceFocalLoss
import torch
import random


batch_size=6
num_epoch=1000
device = 'cuda:0'

seg_list, rl_list, holdout_list = data_spilt('/home/xiangcen/RLModality/picai_h5', 110, 100, 10)
seg_list_promise, rl_list_promise, holdout_list_promise = data_spilt('/home/xiangcen/RLModality/promise_h5', 231, 180, 20)

train_list = rl_list + rl_list_promise
test_list = holdout_list + holdout_list_promise


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
model = RewardModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



num_train_loop = 9999
num_train_patient = 4
num_test_patient = 4
for i in range(num_train_loop):
    train_dir_list = random.sample(train_list, num_train_patient)
    test_dir_list = random.sample(test_list, num_test_patient)
    train_loss = train_reward_model(model, seg_model, train_dir_list, optimizer, torch.nn.BCELoss(), device=device)
    test_acc = test_reward_model(model, seg_model, test_dir_list, device=device)



    print(f'epoch {i}, loss: {train_loss}, acc: {test_acc}')
    
    
    loss_list.append(torch.tensor([train_loss, test_acc]))
    loss_tensor = torch.stack(loss_list)
    torch.save(loss_tensor, '/home/xiangcen/RLModality/models/loss/train_loss_weak.pt')
    torch.save(model.state_dict(), '/home/xiangcen/RLModality/models/weak.ptm')
    print('model saved!')
