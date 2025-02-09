
import torch
from itertools import combinations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RL.Env import Env


# def reward_model_loss(good, bad):
#     difference = torch.sigmoid(good - bad)
#     return -torch.log(difference).mean()

def reward_model_loss(good, bad):
    return -torch.nn.functional.logsigmoid(good - bad).mean()

def sort_state_accuracy(state_list, accuracy_list):
        # Combine the lists into pairs
    paired_list = zip(state_list, accuracy_list)
    ranked_list = sorted(paired_list, key=lambda x: x[1], reverse=True)
    return ranked_list # a list of tuple


def get_rm_list(state_list, accuracy_list):
    ranked_list = sort_state_accuracy(state_list, accuracy_list) # a list of tuple [(state, accuracy), ...]
    all_combinations = list(combinations(ranked_list, r=2)) #[((state, accuracy), (state, accuracy))
                                                            #                                      ]
    
    
    return [state[0][0] for state in all_combinations], \
        [state[1][0] for state in all_combinations]



def generate_rm_data(patient_dir_list, sample_per_patient, loop_per_patient, seg_model, eps_length=999):
    
    good_list, bad_list = [], []
    for patient in patient_dir_list:
        state_list, accuracy_list = [], []
        env = Env(patient, seg_model, eps_length)
        for _ in range(sample_per_patient):
            state, accuracy = env.get_rand_state_accuracy(loop_per_patient)
            if accuracy not in accuracy_list:
                state_list.append(state)
                accuracy_list.append(accuracy)
        
        _good_list, _bad_list = get_rm_list(state_list, accuracy_list)
        good_list += _good_list
        bad_list += _bad_list
        
    return torch.stack(good_list).float(), torch.stack(bad_list).float()


def generate_rm_data_simple(patient_dir_list, seg_model, eps_length=999):
    
    good_list, bad_list = [], []
    for patient in patient_dir_list:
        state_list, accuracy_list = [], []
        env = Env(patient, seg_model, eps_length)
        

        state, accuracy = env.get_both_state_accuracy()
        state_list.append(state)
        accuracy_list.append(accuracy)
        
        state, accuracy = env.get_t2_state_accuracy()
        state_list.append(state)
        accuracy_list.append(accuracy)
        
        state, accuracy = env.get_hb_state_accuracy()
        state_list.append(state)
        accuracy_list.append(accuracy)
        
        state, accuracy = env.get_null_state_accuracy()
        state_list.append(state)
        accuracy_list.append(accuracy)
        

        _good_list, _bad_list = get_rm_list(state_list, accuracy_list)
        good_list += _good_list
        bad_list += _bad_list
        
    return torch.stack(good_list).float(), torch.stack(bad_list).float()



def train_reward_model(
    reward_model,
    segmentation_model,
    train_loader,
    reward_model_optimizer,
    device='cpu',
):
    reward_model.train()
    reward_model.to(device)


    _loss = 0
    _step = 0
    for batch in train_loader:
        _good_list, _bad_list = [], []
        for patient in batch:
            env = Env(patient, segmentation_model, 999, device)
            
            both_tensor, both_acc = env.get_both_state_accuracy()
            t2_tensor, t2_acc = env.get_t2_state_accuracy()
            hb_tensor, hb_acc = env.get_hb_state_accuracy()
            
            
            
            data = [
                {"name": "both", "tensor": both_tensor, "accuracy": both_acc},
                {"name": "t2", "tensor": t2_tensor, "accuracy": t2_acc},
                {"name": "hb", "tensor": hb_tensor, "accuracy": hb_acc},
            ]
            
            sorted_data = sorted(data, key=lambda x: x["accuracy"], reverse=True)
            

            _good_list += [sorted_data[0]['tensor'], sorted_data[0]['tensor'], sorted_data[1]['tensor']]
            _bad_list += [sorted_data[1]['tensor'], sorted_data[2]['tensor'], sorted_data[2]['tensor']]
            
            
            # _good_list += [sorted_data[0]['tensor'], sorted_data[0]['tensor']]
            # _bad_list += [sorted_data[1]['tensor'], sorted_data[2]['tensor']]
        good_tensor = torch.stack(_good_list, dim=0).to(device).float()
        bad_tensor = torch.stack(_bad_list, dim=0).to(device).float()
        


        good_pred = reward_model(good_tensor)
        bad_pred = reward_model(bad_tensor)
        
        # print(good_pred.shape)
        # [f"{num:.2f}" for num in good_pred.flatten().tolist()]
        print('good: ', [f"{num:.2f}" for num in good_pred.flatten().tolist()])
        print('bad:  ', [f"{num:.2f}" for num in bad_pred.flatten().tolist()])
        
        
        loss = reward_model_loss(good_pred, bad_pred)

        
        loss.backward()
        
        reward_model_optimizer.zero_grad()
        
        _loss+=loss.item()
        _step+=1

        # 0 1
        # 0 2
        # 1 2
    return _loss / _step





def test_reward_model(
    reward_model,
    segmentation_model,
    test_loader,
    device='cpu',
):
    reward_model.train()
    reward_model.to(device)

    
    _loss = 0
    _step = 0
    for batch in test_loader:
        good_tensor, bad_tensor = generate_rm_data_simple(
            batch,
            segmentation_model,
        )
        

        with torch.no_grad():

            _good_tensor = good_tensor.to(device)
            _bad_tensor = bad_tensor.to(device)



            o_good = reward_model(_good_tensor)
            o_bad = reward_model(_bad_tensor)

            loss = reward_model_loss(o_good, o_bad)

            _loss += loss.item()
            _step += 1
        
    return _loss/_step




def generate_rm_test_data(patient_dir, loop_per_patient, seg_model, eps_length=999):
    


    env = Env(patient_dir, seg_model, eps_length)

    state, accuracy = env.get_rand_state_accuracy(loop_per_patient)


    
    return state, accuracy

