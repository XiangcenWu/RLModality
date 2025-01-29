from Training import data_spilt
from Training.training_helper_rm import train_reward_model, test_reward_model
from Training.NetWorks import RewardModel
from monai.networks.nets import SwinUNETR
import torch

from monai.data import (
    Dataset,
    DataLoader,
)


batch_size=6
num_epoch=1000
device = 'cuda:0'

seg_list, rl_list, holdout_list = data_spilt('/home/xiangcen/RLModality/picai_h5', 110, 100, 10)
seg_list_promise, rl_list_promise, holdout_list_promise = data_spilt('/home/xiangcen/RLModality/promise_h5', 231, 180, 20)

train_list = rl_list + rl_list_promise
test_list = holdout_list + holdout_list_promise


train_loader = DataLoader(Dataset(train_list), batch_size=1, shuffle=True, drop_last=True)
test_loader = DataLoader(Dataset(test_list), batch_size=1, shuffle=True, drop_last=False)


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



epoch = 99999


loss_list = []
for i in range(epoch):

    train_loss = train_reward_model(model, seg_model, train_loader, optimizer, device=device)
    test_loss = test_reward_model(model, seg_model, test_loader, device=device)
    print(train_loss, test_loss)



    # print(f'epoch {i}, loss: {train_loss}, acc: {test_acc}')
    
    
    loss_list.append(torch.tensor([train_loss, test_loss]))
    loss_tensor = torch.stack(loss_list)
    torch.save(loss_tensor, '/home/xiangcen/RLModality/models/loss/train_loss_rm.pt')
    
    if test_loss <= loss_tensor[:, 1].min():
        torch.save(model.state_dict(), '/home/xiangcen/RLModality/models/rm.ptm')
        print('model saved!')
