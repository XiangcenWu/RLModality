import torch
from monai.data import (
    Dataset,
    DataLoader,
)
from monai.transforms import (
    Compose
)
import h5py
import os
import random

def data_spilt(base_dir, num_seg, num_rl, num_holdout, seed=325):
    list = os.path.listdir(base_dir)
    random.seed(seed)
    random.shuffle(list)
    return list[:num_seg], list[num_seg:num_seg+num_rl], list[-num_holdout:]

    
    
class ReadH5d():
    def __call__(self, file_name):
        with h5py.File(file_name, 'r') as h5_file:
            print(list(h5_file.keys()))

            t2_tensor = torch.tensor(h5_file['t2'][:])   # Load as numpy array, then convert to torch
            hb_tensor = torch.tensor(h5_file['hb'][:])
            cspca_tensor = torch.tensor(h5_file['gt'][:])
        return {
            't2': t2_tensor,
            'hb': hb_tensor,
            'gt': cspca_tensor
        }



def create_data_loader(data_list, batch_size, drop_last=True, shuffle=True):
    set = Dataset(data_list, transform)
    return DataLoader(set, num_workers=8, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)



if __name__ == "__main__":
    transform = Compose([
        ReadH5d()
    ])
    dict = transform('/home/xiangcen/RLModality/picai_h5/20.h5')
    print(dict['gt'].shape)
