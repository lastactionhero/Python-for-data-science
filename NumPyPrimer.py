#%%
import numpy as np
#Array
lst = list(range(0,10))
ar = np.array(lst)
print(ar)
matr = np.array([[1,2,3],[11,22,33],[111,222,333]])
print(matr)
#use numpy functions 
# %%
np.arange(0,10)
#%%
np.zeros(4)
#%%
np.zeros((4,2))
#%%
np.ones((2,4))
# %%
np.linspace(1,5,5)
# %%
#identity matrix
np.eye(10)
# %%
#random 
np.random.rand(5)

# %%
#normal distribution
np.random.randn(2)

# %%
np.random.randint(10,20,10)

# %%
#reshape array, max, min , max number index, min number index
arr = np.random.randint(10,100,25)
print(arr)

print(arr.reshape(5,5))
print(arr.max())
print(arr.argmax())
print(arr.min())
print(arr.argmin())

# %%
#array indexing
arr = np.arange(0,11)
arr
# %%
arr[1]
np.std
# %%
arr[1:]

# %%
arr[0:2]

# %%
arr[:4]

# %%
#Broadcast assignment
arr[:3]=100
arr

# %%
#Indexing 2d array
arr = np.arange(0,25)
arr=arr.reshape(5,5)
arr
# %%
arr[1][2]

# %% 
# comma notation
arr[1,2]

# %% 
# Slice notation
arr[:3,:4]

# %%
#Get bottom row
arr[-1]
# %%
arr[-1,:] = 100
arr

# %%
#Array comparison
arr = np.arange(1,11)
arr

# %%
arr>5

# %%
arr[arr>5]

# %%
arr[arr>=5]=100
arr
# %%
arr+1000

# %%
#universal functions
arr= np.arange(1,11)
arr=arr**2
arr
# %%
np.sqrt(arr)

# %%
arr = np.arange(1,21).reshape((5,4))
arr

# %%
#columnar sum
arr.sum(axis=0)

# %%
#row wise sum
arr.sum(axis=1)

# %%
