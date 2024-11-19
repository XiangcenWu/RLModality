from Training import data_spilt, ReadH5d, create_data_loader
from Training import train_weak_net, test_weak_net
from Training.NetWorks import WeakModel
from monai.transforms import *
from monai.networks.nets import DynUNet, SwinUNETR
from monai.losses import DiceFocalLoss
import torch
batch_size=6
num_epoch=1000
device = 'cuda:0'


seg_list, rl_list, holdout_list = data_spilt('/home/xiangcen/RLModality/picai_h5', 190, 0, 20)


train_transform = ReadH5d()

inference_transform = ReadH5d()
train_loader = create_data_loader(seg_list, train_transform, batch_size=batch_size, shuffle=True)
inference_loader = create_data_loader(rl_list+holdout_list, inference_transform, batch_size=batch_size, shuffle=True)

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
model = WeakModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_list = []
for i in range(1000):
    train_loss = train_weak_net(model, seg_model, train_loader, optimizer, torch.nn.BCELoss(), device=device)
    test_acc = test_weak_net(model, seg_model, inference_loader, device=device)



    print(f'epoch {i}, loss: {train_loss}, acc: {test_acc}')
    
    
    loss_list.append(torch.tensor([train_loss, test_acc]))
    loss_tensor = torch.stack(loss_list)
    torch.save(loss_tensor, '/home/xiangcen/RLModality/models/loss/train_loss_weak.pt')
    torch.save(model.state_dict(), '/home/xiangcen/RLModality/models/weak.ptm')
    print('model saved!')
