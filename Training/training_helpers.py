import torch
import random




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
    
functions = [get_both, get_hb, get_t2]
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





def generate_rand_index(max: int = 32):
    num_of_generations = torch.randint(1, 33, (1, )).item()  # Random number of generations between 1 and 32
    
    # Generate random indices without replacement
    _indexs = torch.randperm(max)  # Generate random permutation of indices [0, 1, 2, ..., max-1]
    
    # Select num_of_generations number of random indices
    sampled_indexes = _indexs[:num_of_generations]
    
    # Rank the sampled indexes (sorting them in ascending order)
    # sampled_indexes = torch.sort(sampled_indexes)[0]  # Get the indices that would sort the tensor
    
    return sampled_indexes

def train_weak_net(
    weak_model,
    seg_model, 
    weak_loader,
    weak_optimizer,
    loss_function,
    device='cpu',
):
    weak_model.train()
    weak_model.to(device)
    seg_model.eval()
    seg_model.to(device)
    
    step = 0.
    loss_a = 0.
    for batch in weak_loader:
        
        if random.random() < 0.2: # implement all radiologist annotation
            
            label = batch['gt']
            both = get_both(batch)
            img = torch.cat([both, label], dim=1).to(device)
            
            batch_size = img.shape[0]
            label = torch.ones(size=(batch_size, 1)).to(device)
            
            output = weak_model(img.float())
            loss = loss_function(output, label)
            loss.backward()
            weak_optimizer.step()
            weak_optimizer.zero_grad()
            
            loss_a+= loss.item()
            
        else:
            # random generate index and mask the index
            index_to_mask = generate_rand_index()
            label = batch['gt'].to(device)
            label[..., index_to_mask] = 0
            
            
            with torch.no_grad():
                selected_function = random.choice(functions)
                selected_modality = selected_function(batch).to(device)
                output = seg_model(selected_modality)
                output[..., index_to_mask] = 0 # mask the output as well
                dice_score = dice_coef_batch(output, label).to(device)
                
            both = get_both(batch).to(device)
            img = torch.cat([both, output], dim=1).to(device)

        
        
        
            output = weak_model(img)
            loss = loss_function(output, dice_score)
            loss.backward()
            weak_optimizer.step()
            weak_optimizer.zero_grad()
            
            loss_a+= loss.item()


        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch


def test_weak_net(
    weak_model,
    seg_model, 
    weak_loader,
    device='cpu',
):
    weak_model.eval()
    weak_model.to(device)
    seg_model.eval()
    seg_model.to(device)
    
    step = 0.
    difference_a = 0.
    for _ in range(30):
        for batch in weak_loader:
            # random generate index and mask the index
            index_to_mask = generate_rand_index()
            label = batch['gt'].to(device)
            label[..., index_to_mask] = 0
            
            
            
            with torch.no_grad():
                selected_function = random.choice(functions)
                selected_modality = selected_function(batch).to(device)
                output = seg_model(selected_modality)
                output[..., index_to_mask] = 0 # mask the output as well
                dice_score = dice_coef_batch(output, label).to(device)
                
                both = get_both(batch).to(device)
                img = torch.cat([both, output], dim=1).to(device)

                output = weak_model(img)
                difference_a += torch.abs(output - dice_score).mean().item()

            step+=1
    return difference_a / step



def test_single_weak_net(
    weak_model,
    seg_model, 
    weak_loader,
    device='cpu',
):
    weak_model.eval()
    weak_model.to(device)
    seg_model.eval()
    seg_model.to(device)
    

    difference_list = []
    for _ in range(100):
        for batch in weak_loader:
            # random generate index and mask the index
            index_to_mask = generate_rand_index()
            label = batch['gt'].to(device)
            label[..., index_to_mask] = 0
            
            
            
            with torch.no_grad():
                selected_function = random.choice(functions)
                selected_modality = selected_function(batch).to(device)
                output = seg_model(selected_modality)
                output[..., index_to_mask] = 0 # mask the output as well
                dice_score = dice_coef_batch(output, label).to(device)
                
                both = get_both(batch).to(device)
                img = torch.cat([both, output], dim=1).to(device)

                output = weak_model(img)
                difference_list.append(torch.abs(output - dice_score).mean().item())
                print(f'output {output.item()}, dice {dice_score.item()}')
    _tensor = torch.tensor(difference_list)
    print(_tensor)
    return _tensor.mean(), _tensor.std()








def test_single_weak_net_with_continouw_index(
    weak_model,
    seg_model, 
    weak_loader,
    device='cpu',
):
    weak_model.eval()
    weak_model.to(device)
    seg_model.eval()
    seg_model.to(device)
    

    difference_list = []
    for _ in range(100):
        for batch in weak_loader:
            # random generate index and mask the index
            index_to_mask = [0, 1, 2, 3, 4,  6, 7, 8, 9, 10, 11,12, 13, 14, 15, 16, 17, 18,19,20,21,22,23,24,25,26,27,28,29,30, 31]
            index_to_mask = torch.arange(12, 13).tolist()
            label = batch['gt'].to(device)
            label[..., index_to_mask] = 0
            
            
            
            with torch.no_grad():
                selected_function = random.choice(functions)
                selected_modality = selected_function(batch).to(device)
                output = seg_model(selected_modality)
                output[..., index_to_mask] = 0 # mask the output as well
                dice_score = dice_coef_batch(output, label).to(device)
                
                both = get_both(batch).to(device)
                img = torch.cat([both, output], dim=1).to(device)

                output = weak_model(img)
                difference_list.append(torch.abs(output - dice_score).mean().item())
                print(f'output {output.item()}, dice {dice_score.item()}')
    _tensor = torch.tensor(difference_list)
    print(_tensor)
    return _tensor.mean(), _tensor.std()


if __name__ == "__main__":
    img = torch.rand(128, 128)
    batch_size = img.shape[0]
    print(batch_size)