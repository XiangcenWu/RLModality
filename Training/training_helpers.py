import torch





def post_process(x: torch.tensor, th: int=0.5):
    return (x > th).to(float)

def dice_coefficient(prediction: torch.tensor, gt: torch.tensor, post=False):
    assert prediction.shape == gt.shape, "Masks must be the same shape"
    if post:
        prediction = post_process(prediction)
    return (2 * torch.sum(prediction * gt)) / (torch.sum(prediction) + torch.sum(gt))


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
    

def train_seg_net(
        seg_model, 
        seg_loader,
        seg_optimizer,
        seg_loss_function,
        device='cpu',
    ):
    seg_model.train()
    
    step = 0.
    loss_a = 0.
    for batch in seg_loader:
        img, label = batch["image"].to(device), batch["label"].to(device)
        # forward pass and calculate the selection
        # forward pass of selected data
        output = seg_model(img)
        loss = seg_loss_function(output, label)
        loss.backward()
        seg_optimizer.step()
        seg_optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch



def test_sel_net_h5(
        sel_model, 
        sel_loader,
        query_set_true_mean,
        query_set_true_std,
        sel_loss_function,
        seg_model,
        device='cpu',
    ):
    seg_model.eval()
    sel_model.eval()
    
    
    step_e = 0.
    loss_e = 0.
    for batch in sel_loader:
        img, label = batch["image"].to(device), batch["label"].to(device)
        with torch.no_grad():
            output = seg_model(img)
            accuracy = dice_coefficient(output, label, post=True)
            

        loss_e += loss.item()
        step_e += 1
    return loss_e / step_e



if __name__ == "__main__":
    x = torch.ones(2, 1, 128, 128, 32)
    y = torch.rand_like(x)




    print(dice_coefficient(x, y, post=True))