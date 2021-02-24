#%%
import matplotlib.pyplot as plt
import numpy as np

#%%matplotlib inline

# %%
x = np.linspace(0,5,50)
y = x**2
# %%
plt.plot(x,y,'r')
# %%
# functional way of plotting multiple plots
plt.subplot(1,2,1)
plt.plot(x,y,'r')
plt.subplot(1,2,2)
plt.plot(y,x,'g')

# %%
#get imaginary canvas
fig=plt.figure()
axes = fig.add_axes([0.8,0.8,0.8,0.8])
axes.plot(x,y,'g')
axes.plot(y,x,'r')
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')

# %%
fig = plt.figure()
axis1 = fig.add_axes([0.1,0.1,0.9,0.9])
axis2 = fig.add_axes([0.3,0.6,0.3,0.3])

# %%
axis1.plot(x,y)
axis2.plot(y,x)
fig
# %%
fig, axes = plt.subplots(nrows=1,ncols=2)
for curaxis in axes:
    curaxis.plot(x,y)
# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,2))
axes[0].plot(x,y)
axes[1].plot(y,x)

# %%
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.plot(x,y,color="green", linewidth=2, alpha=0.8,linestyle='-'
, marker ='o', markersize=1)

# %%
#set limit over x axis
axes.set_xlim([0,0.2])
axes.set_ylim([0,0.2])
fig
# %%
