import os
import random
from typing import Tuple

def spilt_data(data_dir: str, num_seg_train: int, num_agent_train: int, num_agent_test:int , seed: int=325) -> Tuple[int, int, int]:
    data_list = os.listdir(data_dir)
    data_list = list(map(lambda x: os.path.join(data_dir, x), data_list))
    
    random.seed(seed)
    random.shuffle(data_list)

    seg_train = data_list[: num_seg_train]
    agent_train = data_list[num_seg_train : num_seg_train+num_agent_train]
    agent_test = data_list[num_seg_train+num_agent_train :]
    
    
    return seg_train, agent_train, agent_test