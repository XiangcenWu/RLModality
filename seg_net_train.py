from DataLoading import *
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss, GeneralizedDiceFocalLoss, HausdorffDTLoss
from Training import train_seg_net, test_seg_net
import pickle
from monai.transforms import *
# spilt the dataset
seg_list_picai, agent_train_picai, agent_test_picai = spilt_data('picai_miccai_data_human', 70, 120, 30)
seg_list_pormise, agent_train_pormise, agent_test_pormise = spilt_data('promise_miccai-new', 151, 220, 60)

seg_list = seg_list_picai + seg_list_pormise
agent_train = agent_train_picai + agent_train_pormise
agent_test = agent_test_picai + agent_test_pormise


train_trainform = ([
    ReadH5d(),
    RandAffined(['t2','hb', 'gt'], spatial_size=(128, 128, 64), prob=0.25, shear_range=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1), mode='nearest', padding_mode='zeros'),
    RandGaussianSmoothd(['t2','hb'], prob=0.1),
    RandGaussianNoised(['t2','hb'], prob=0.1, std=0.05),
    RandAdjustContrastd(['t2','hb'], prob=0.1, gamma=(0.5, 2.))
    
])
train_loader = get_loader(seg_list, ReadH5d(), 4, True, False)
test_loader = get_loader(agent_test+agent_train, ReadH5d(), 1, True, True)
model = SwinUNETR(
    (128, 128, 64),
    2,
    1,
    drop_rate= 0.1,
    attn_drop_rate= 0.1,
    dropout_path_rate= 0.1,
    downsample='mergingv2',
    use_v2=True
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_function = GeneralizedDiceFocalLoss(include_background=True, to_onehot_y=False, sigmoid=True)



result = []
for i in range(1000):
    loss = train_seg_net(
        model=model,
        train_loader=train_loader,
        train_optimizer=optimizer,
        train_loss=loss_function,
        both=0.33,
        hb=0.33,
        t2=0.34,
        device='cuda:0'
    )
    t2_dice, hb_dice, both_dice = test_seg_net(
        model=model,
        test_loader=test_loader,
        device='cuda:0'
    )
    print('*********************************')
    print(f'epoch {i}, loss: {loss}')
    print(f't2 dice: {t2_dice}, hb dice: {hb_dice}, both dice: {both_dice}')
    result.append([loss, t2_dice, hb_dice, both_dice])
    tensor_result = torch.stack([torch.tensor(_, dtype=float) for _ in result])
    with open('train_loss.pkl', 'wb') as file:
        pickle.dump(tensor_result, file)
        
    torch.save(model.state_dict(), './models/T2HbSeg.ptm')
    print('model saved!')