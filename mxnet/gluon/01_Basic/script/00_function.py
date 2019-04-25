# %%
import os
import numpy as np
from mxnet import nd
from matplotlib import pyplot as plt

# %%
# Array
a = np.array([[2, 2], [1,2]])
b = nd.array([[2, 2], [1,2]])
print('====== Numpy ======')
print(a)
print('====== MXNet ======')
print(b)

# %%
# Ones
a = np.ones((2, 2))
b = nd.ones((2, 2))
print('====== Numpy ======')
print(a)
print('====== MXNet ======')
print(b)

#%%
# Normal Distribution
a = np.random.normal(0, 1, (2, 2))
b = nd.random.normal(0, 1, (2, 2))
print('====== Numpy ======')
print(a)
print('====== MXNet ======')
print(b)
#%%
# Full
a = np.full((2,2), 5)
b = nd.full((2,2), 5)
print('====== Numpy ======')
print(a)
print('====== MXNet ======')
print(b)

#%%
# Operation 1
a = np.full((2,2), 5)
b = nd.full((2,2), 5)
print('====== Numpy ======')
print(a*3)
print('====== MXNet ======')
print(b*3)

#%%
# Operation 2
a1 = np.random.normal(0, 1, (2,1))
a2 = np.random.normal(0, 1, (1,2))

b1 = nd.random.normal(0, 1, (2,1))
b2 = nd.random.normal(0, 1, (1,2))
print('====== Numpy ======')
print(a1*a2.T)
print('====== MXNet ======')
print(b1*b2.T)

#%%
# Operation 3
a1 = np.random.normal(0, 1, (2,1))
a2 = np.random.normal(0, 1, (1,2))

b1 = nd.random.normal(0, 1, (2,1))
b2 = nd.random.normal(0, 1, (1,2))
print('====== Numpy ======')
print(np.exp(a2.T * a1))
print('====== MXNet ======')
print((b2.T * b1).exp())

#%%
# Operation 2
a1 = nd.random.normal(0, 1, (2,1))
a2 = nd.random.normal(0, 1, (1,2))

print('====== MXNet ======')
print((b2.T * b1).exp())
print('====== Numpy ======')
print((b2.T * b1).exp().asnumpy())
