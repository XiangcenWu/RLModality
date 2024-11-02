import h5py
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import Transform

def get_loader(
        list, 
        transform, 
        batch_size: int,
        shuffle: bool, 
        drop_last: bool, 
    ):
    _ds = Dataset(list, transform=transform)

    return DataLoader(
        dataset = _ds,
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last
    )



class ReadH5d(Transform):

    def __init__(self) -> None:
        super().__init__()


    def __call__(self, file_dir: str) -> dict:
        return self.get_training_dict(file_dir=file_dir)
    
    
    
    def get_training_dict(self, file_dir: str) -> dict:
        with h5py.File(file_dir, 'r') as h5_file:
            # Load each dataset and convert to torch tensors if needed
            t2_tensor = torch.tensor(h5_file['t2_tensor'][:])   # Load as numpy array, then convert to torch
            hb_tensor = torch.tensor(h5_file['hb_tensor'][:])
            cspca_tensor = torch.tensor(h5_file['cspca_tensor'][:])
            
            
        return {
            't2': t2_tensor ,
            'hb': hb_tensor ,
            'gt': cspca_tensor ,
        }
        
