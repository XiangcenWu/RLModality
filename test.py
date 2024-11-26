index_128 = [slice(0, 43), slice(43, 86), slice(86, 128)]
index_32 = [slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16), slice(16, 20), slice(20, 24), slice(24, 28), slice(28, 32)]

import torch

x = torch.rand(1, 128, 128, 32)




# Select [:, 0:32, 32:64, 16:24]
y = x[:, index_128[0], index_128[1], index_32[7]]


print(y.shape)