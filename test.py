import torch
from torch.distributions.categorical import Categorical





x = torch.tensor([
    [0.2, 0.8],
    [0.6, 0.4]
])



c = Categorical(x)


action = torch.tensor([
    [0],
    [1]
])


print(c.log_prob(action))