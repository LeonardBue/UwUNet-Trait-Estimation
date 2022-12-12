# %%
# short script to scale an image of reconstruction and original to same value 
# range for visualization of an RGB representation
# initialize
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
from sklearn.preprocessing import MinMaxScaler

model_name = 'EO1H0450222004225110PZ'
FW = 5.7885 # text width in Latex

# %%
# get scaler
with rasterio.open('D:/Projects/MA/data/Hyperion/cropped/reconstruction_clipped.tif') as img:
    r = img.read(16)
    g = img.read(5)
    b = img.read(1)
    rgb = np.dstack((r,g,b))

scaler = MinMaxScaler()
scaler.fit(rgb.reshape(-1, 3))

# %%
# scale and plot
path_pred = 'data/Hyperion/'+model_name+'/results_spatial/'
path_or = f'./data/Hyperion/cropped/{model_name}_{14}.tif'
path_re = os.path.join(path_pred, 'test', model_name, f'prediction_{14}.tif')

with rasterio.open(path_re) as img:
    r = img.read(16)
    g = img.read(5)
    b = img.read(1)
    arr = np.dstack((r,g,b))
    profile_re = img.profile
    profile_re.update(driver='GTiff')

with rasterio.open(path_or) as img:
    r = img.read(16)
    g = img.read(5)
    b = img.read(1)
    rgb = np.dstack((r,g,b))
    profile_or = img.profile
    profile_or.update(driver='GTiff')

arr = scaler.transform(arr.reshape(-1, 3)).reshape(arr.shape)
arr[g==0.,:] = 0
rgb = scaler.transform(rgb.reshape(-1, 3)).reshape(rgb.shape)
rgb[g==0.,:] = 0

fig1, ax1 = plt.subplots(figsize=(0.6*FW, 0.3*FW))
ax1.imshow(arr)
ax1.set_xticks(np.arange(arr.shape[1]+1, step=int(arr.shape[1]/4)))
ax1.set_yticks(np.arange(arr.shape[0]+1, step=int(arr.shape[0]/4)))
ax1.grid(False)

fig2, ax2 = plt.subplots(figsize=(0.6*FW, 0.3*FW))
ax2.imshow(rgb)
ax2.set_xticks(np.arange(rgb.shape[1]+1, step=int(rgb.shape[1]/4)))
ax2.set_yticks(np.arange(rgb.shape[0]+1, step=int(rgb.shape[0]/4)))
ax2.grid(False)

# %%
# save
with rasterio.open(os.path.join(path_pred, 'test', model_name, f'reconstruction_{14}.tif'), 'w', **profile_re) as im_re:
    for l in range(arr.shape[2]):
        im_re.write(arr[...,l], l+1)

with rasterio.open(f'./data/Hyperion/cropped/{model_name}_{14}_scaled.tif', 'w', **profile_or) as im_or:
    for l in range(rgb.shape[2]):
        im_or.write(rgb[...,l], l+1)

# %%
