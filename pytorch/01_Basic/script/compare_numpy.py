# Compare with numpy
# %% 
import numpy as np
import torch
# %%
a = np.array([1, 2, 3, 4])
b = torch.Tensor([1,2,3,4])

print(a.sum(), b.sum())

# %%
# BroadCasting
n1 = np.array([[2, 8]])
n2 = np.array([[1], [3]])
print(n1*n2)

b1 = torch.Tensor([[2, 8]])
b2 = torch.Tensor([[1], [3]])
print(b1*b2)
