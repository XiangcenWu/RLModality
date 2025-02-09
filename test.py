import torch




x = torch.rand(size=(3, 5))
print(x)

index = torch.tensor([0, 1, 1])
selected = x.gather(dim=1, index=index.unsqueeze(1))



print(selected)


import math
print(math.log(0.0002533))