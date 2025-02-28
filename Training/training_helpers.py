import torch
import random
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def post_process(x: torch.tensor, th: int=0.5):
    return (x > th).to(float)

def dice_coefficient(prediction: torch.tensor, gt: torch.tensor, post=False):
    assert prediction.shape == gt.shape, "Masks must be the same shape"
    if post:
        prediction = post_process(prediction)
    return (2 * torch.sum(prediction * gt)) / (torch.sum(prediction) + torch.sum(gt) + 1e-6)


def dice_coef_batch(prediction: torch.tensor, gt: torch.tensor):
    batch_size = prediction.shape[0]
    assert prediction.shape == gt.shape, "Masks must be the same shape"
    _tensor = torch.zeros(size=(batch_size, 1))
    for i in range(batch_size):
        _tensor[i, 0] = dice_coefficient(prediction[i], gt[i], post=True)
    return _tensor


def get_t2(batch):
    t2 = batch['t2']
    return torch.cat([t2, torch.zeros_like(t2)], dim=1)


def get_hb(batch):
    hb = batch['hb']
    return torch.cat([torch.zeros_like(hb), hb], dim=1)


def get_both(batch):
    t2 = batch['t2']
    hb = batch['hb']
    return torch.cat([t2, hb], dim=1)
    
functions = [get_both, get_hb, get_t2, get_t2]
def train_seg_net(
        seg_model, 
        seg_loader,
        seg_optimizer,
        seg_loss_function,
        device='cpu',
    ):
    seg_model.train()
    seg_model.to(device)
    
    step = 0.
    loss_a = 0.
    for batch in seg_loader:
        label = batch['gt'].to(device)
        selected_function = random.choice(functions)
        img = selected_function(batch).to(device)

        
        
        
        output = seg_model(img)
        loss = seg_loss_function(output, label)
        loss.backward()
        seg_optimizer.step()
        seg_optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch



def test_seg_net(
        seg_model, 
        seg_loader,
        device='cpu',
    ):
    seg_model.eval()
    seg_model.to(device)
    
    
    step_e = 0.
    loss_both = 0.
    loss_t2 = 0.
    loss_hb = 0.
    
    
    
    for batch in seg_loader:
        label = batch["gt"].to(device)
        both = get_both(batch).to(device)
        t2 = get_t2(batch).to(device)
        hb = get_hb(batch).to(device)
        with torch.no_grad():
            output = seg_model(both)
            loss_both += dice_coefficient(output, label, post=True)
            
            output = seg_model(t2)
            loss_t2 += dice_coefficient(output, label, post=True)

            output = seg_model(hb)
            loss_hb += dice_coefficient(output, label, post=True)



        step_e += 1

    return loss_both / step_e, loss_t2 / step_e, loss_hb / step_e




def test_seg_net_ver2(
        seg_model, 
        seg_loader,
        device='cpu',
    ):
    seg_model.eval()
    seg_model.to(device)
    
    
    step_e = 0.
    loss_both = 0.
    loss_t2 = 0.
    loss_hb = 0.
    
    
    t2_list, hb_list, both_list = [], [], []
    for batch in seg_loader:
        label = batch["gt"].to(device)
        both = get_both(batch).to(device)
        t2 = get_t2(batch).to(device)
        hb = get_hb(batch).to(device)
        with torch.no_grad():
            output = seg_model(both)
            both_list.append(dice_coefficient(output, label, post=True).item())
            
            output = seg_model(t2)
            t2_list.append(dice_coefficient(output, label, post=True).item())

            output = seg_model(hb)
            hb_list.append(dice_coefficient(output, label, post=True).item())



        
    _t2 = torch.tensor(t2_list)
    _hb = torch.tensor(hb_list)
    _both = torch.tensor(both_list)
    print(_t2.mean(), _t2.std(), _hb.mean(), _hb.std(), _both.mean(), _both.std())
