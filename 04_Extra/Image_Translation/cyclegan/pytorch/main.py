# %%
import torch
from torch import nn

from models import *
# %%
test = torch.empty(1, 3, 224, 224)
G_A = Generator(3, 3, 64, "IN", 2, 5)
# G_B = Generator(3, 3, 64, "IN", 2, 5)

D_A = Discriminator(3, 64, "BN", 4)
# D_B = Discriminator(3, 64, "BN", 4)

result_G = G_A(test)
result_D = D_A(test)

print(test.shape, result_G.shape, result_D.shape)