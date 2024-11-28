
import torch

x = torch.randint(0, 10, size=(8, )).view(2, 4)

print(x)

print(torch.chunk(x, chunks=4, dim=1))

