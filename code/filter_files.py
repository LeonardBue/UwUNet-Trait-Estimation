#%% Manually filter images (visual assessment of good atmospheric correction)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import spectral.io.envi as envi

scale_factor = 1000
dir_r = 'data/Hyperion/'
df_fnames = pd.read_csv(dir_r+'filenames.csv')
fnames = df_fnames['fnames'].values
keeps=[]


for fname in fnames:
    try:
        img=envi.open(dir_r+fname+'_Reflectance_topo_v2_Reflectance.hdr')
        bands=np.asarray(img.metadata['wavelength']).astype(float)
        img=rio.open(dir_r+fname+'_Reflectance_topo_v2_Reflectance.img')
    except:
        img=envi.open(dir_r+fname+'_L1R_complete.hdr')
        bands=np.asarray(img.metadata['wavelength']).astype(float)
        img=rio.open(dir_r+fname+'_L1R_complete.img')
    R=img.read(int(np.argmin(np.abs(bands-640))+1))
    G=img.read(int(np.argmin(np.abs(bands-560))+1))
    B=img.read(int(np.argmin(np.abs(bands-457))+1))
    RGB = np.stack([R, G, B], axis=2)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(RGB/scale_factor)
    plt.tight_layout()
    plt.show(block=False)

    keep=input('keep '+fname+' ? [y/n]: ')
    if keep=='n':
        print(fname+' not kept')
        keeps.append(False)
    elif keep=='y':
        print(fname+' kept')
        keeps.append(True)
    plt.close()

data={'fnames':fnames,'keep':keeps}
df_toKeep=pd.DataFrame.from_dict(data)
# df_toKeep.to_csv(dir_r+'images_visual_check.csv')
# %% Show images that are not kept
fig, axes = plt.subplots(1, len(keeps[keeps['keep']==False]['keep']), figsize =(20, 10))
for i, fname in enumerate(keeps[keeps['keep']==False]['fnames'].values):
    img=envi.open(dir_r+fname+'_Reflectance_topo_v2_Reflectance.hdr')
    bands=np.asarray(img.metadata['wavelength']).astype(float)
    img=rio.open(dir_r+fname+'_Reflectance_topo_v2_Reflectance.img')
    R=img.read(int(np.argmin(np.abs(bands-640))+1))
    G=img.read(int(np.argmin(np.abs(bands-560))+1))
    B=img.read(int(np.argmin(np.abs(bands-457))+1))
    RGB = np.stack([R, G, B], axis=2)
    axes[i].imshow(RGB/scale_factor)
plt.tight_layout()
plt.show()

# %%
