import torch
import random
from monai.metrics import DiceMetric


def train_seg_net(
        model, 
        train_loader,
        train_optimizer,
        train_loss,
        both=0.33,
        hb=0.34,
        t2=0.33,
        device='cpu',

    ):
    # use random to determine which modalidy to train
    # List of conditions (as functions)
    conditions = [get_t2_image, get_hb_image, get_both]
    # Define the probabilities for each condition
    probabilities = [t2, hb, both]  # Total should sum to 1
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.train()
    
    _step = 0.
    _loss = 0.
    for batch in train_loader:

        # Randomly select a condition based on the probabilities
        selected_condition = random.choices(conditions, weights=probabilities, k=1)[0]

        # Call the selected condition function
        img, gt = selected_condition(batch, device)




        # forward pass of selected data
        output = model(img)
        
        loss = train_loss(output, gt)
        # print(loss)

        

        loss.backward()
        train_optimizer.step()
        train_optimizer.zero_grad()

        _loss += loss.item()
        _step += 1.
    _epoch_loss = _loss / _step

    return _epoch_loss



def get_t2_image(batch, device):
    t2, _, gt = batch['t2'], batch['hb'], batch['gt']

    img = torch.cat([t2, torch.zeros_like(t2)], dim=1)

    return img.to(device), gt.to(device)



def get_hb_image(batch, device):
    _, hb, gt = batch['t2'], batch['hb'], batch['gt']
    
    img = torch.cat([torch.zeros_like(hb), hb], dim=1)

    return img.to(device), gt.to(device)


def get_both(batch, device):
    t2, hb, gt = batch['t2'], batch['hb'], batch['gt']
    img = torch.cat([t2, hb], dim=1)
    return img.to(device), gt.to(device)








def dice_coefficient(pred: torch.Tensor, gt: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Calculate the Dice coefficient between two binary tensors.

    Args:
        pred (torch.Tensor): Predicted binary tensor (output of a model).
        target (torch.Tensor): Ground truth binary tensor.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    # Flatten the tensors to ensure they have the same shape
    pred = pred.view(-1)
    gt = gt.view(-1)

    # Calculate intersection and union
    intersection = (pred * gt).sum()
    dice = (2.0 * intersection + epsilon) / (pred.sum() + gt.sum() + epsilon)
    
    return dice.item()  # Convert to float


def PostInference(prediction: torch.tensor):
    prediction = torch.sigmoid(prediction)
    prediction = prediction > 0.5
    prediction.to(float)
    return prediction


def dice_metric(prediction, gt):
    
    prediction = PostInference(prediction)
    
    return dice_coefficient(prediction, gt)


def test_seg_net(
        model, 
        test_loader,
        device='cpu',
    ):

    model.to(device)
    model.eval()
    
    step = 0.
    t2_dice = 0.
    hb_dice = 0.
    both_dice = 0.
    for batch in test_loader:


        with torch.no_grad():
            # Call the selected condition function
            img_t2, _ = get_t2_image(batch, device)
            img_hb, _ = get_hb_image(batch, device)
            img_both, gt = get_both(batch, device)




            # forward pass of selected data
            output_t2 = model(img_t2)
            output_hb = model(img_hb)
            output_both = model(img_both)
            
            _t2_dice = dice_metric(output_t2, gt)
            _hb_dice = dice_metric(output_hb, gt)
            _both_dice = dice_metric(output_both, gt)

            t2_dice += _t2_dice
            hb_dice += _hb_dice
            both_dice += _both_dice
            step += 1.
    

    return t2_dice/step, hb_dice/step, both_dice/step