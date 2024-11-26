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


train_transform = Compose([
    ReadH5d(),
    RandAffined(['t2','hb', 'gt'], spatial_size=(128, 128, 32), prob=0.25, shear_range=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1), mode='nearest', padding_mode='zeros'),
    RandGaussianSmoothd(['t2','hb'], prob=0.1),
    RandGaussianNoised(['t2','hb'], prob=0.1, std=0.05),
    RandAdjustContrastd(['t2','hb'], prob=0.1, gamma=(0.5, 2.))
])
inference_transform = ReadH5d()
train_loader = create_data_loader(seg_list+seg_list_promise, train_transform, batch_size=batch_size, shuffle=True)
inference_loader = create_data_loader(rl_list + holdout_list+rl_list_promise + holdout_list_promise, inference_transform, batch_size=1, shuffle=False)



model = SwinUNETR(
    img_size = (128, 128, 32),
    in_channels = 2,
    out_channels = 1,
    depths = (2, 2, 2, 2),
    num_heads = (3, 6, 12, 24),
    # drop_rate = 0.1,
    # attn_drop_rate = 0.1,
    # dropout_path_rate = 0.1,
    downsample="mergingv2",
    use_v2=True,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



loss_list = []
for i in range(800):
    train_loss = train_seg_net(model, train_loader, optimizer, DiceFocalLoss(sigmoid=True), device=device)
    both_acc, t2_acc, hb_acc = test_seg_net(model, inference_loader, device=device)

    print(f'epoch {i}, loss {train_loss} \n both {both_acc}, t2 {t2_acc}, hb {hb_acc}')


    loss_list.append(torch.tensor([train_loss, both_acc, t2_acc, hb_acc]))
    loss_tensor = torch.stack(loss_list)
    torch.save(loss_tensor, '/home/xiangcen/RLModality/models/loss/train_loss.pt')
    torch.save(model.state_dict(), '/home/xiangcen/RLModality/models/segmentation.ptm')
    print('model saved!')
