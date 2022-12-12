# Visualize real raw and filtered, and synthetic reflctance spectrum
# %%
# Read images and extract bands
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('./code')
from utils import filter_array


cmap = sns.color_palette("mako", as_cmap=True)
palette = sns.color_palette("crest_r", 20)
palette_diverging = sns.color_palette("PuOr_r", 11)
cmap_diverging = sns.color_palette("icefire", 11, as_cmap=True)
sns.set()
sns.set_style("whitegrid")
sns.despine()
sns.set_context("paper", rc={"font.size":11,"axes.titlesize":11,"axes.labelsize":11})

dir_h = 'data/Hyperion/'
dir_s = 'data/dbs/'
image_ids = 'data/Hyperion/images_visual_check.csv'
df_fnames = pd.read_csv(image_ids)
fnames = df_fnames[df_fnames['keep']==True]['fnames'].values
fname = 'EO1H0450222004225110PZ'
sp_class = {   'conifers':[1,2,3,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,30,32,33,34,35,36],
            'broadleafes':[4,5,6,7,8,9,10,12,27,28,29,31,37]}

# shifted by number of additional input bands (14)
shift = 14 # omit first 14 bands as they are not reflectance values
scale_factor = 10000. # scaling necessary for Hyperion data due to data type

FW = 5.7885 # text width in Latex in inches

# %%
# data preparation
with rasterio.open(dir_h+fname+'_Reflectance_topo_v2_Reflectance.img') as img_h:
    ar_h = np.float32(img_h.read())
ar_h /= scale_factor
ar_h[ar_h > 1.] = 0.
ar_h,_ = filter_array(ar_h, theta_cc=0.6, theta_conf=0.5, theta_lc=[210,220], lct_sp=sp_class)
ar_h[ar_h == 0.] = np.nan
ar_h = ar_h.reshape((ar_h.shape[0], 1, -1)).squeeze()
mu_h = np.nanmean(ar_h, axis=1)
# s_h = np.nanstd(ar_h, axis=1)
min_h = np.nanmin(ar_h, axis=1)
max_h = np.nanmax(ar_h, axis=1)
del(ar_h)

with rasterio.open(dir_h+'cropped/'+'mosaic.tif') as img_hf:
    ar_hf = np.float32(img_hf.read())
ar_hf,_ = filter_array(ar_hf, theta_cc=0.6, theta_conf=0.5, theta_lc=[210,220], lct_sp=sp_class)
ar_hf[ar_hf == 0.] = np.nan
ar_hf = ar_hf.reshape((ar_hf.shape[0], 1, -1)).squeeze()
mu_hf = np.nanmean(ar_hf, axis=1)
# s_hf = np.nanstd(ar_hf, axis=1)
min_hf = np.nanquantile(ar_hf, 0.005, axis=1)
max_hf= np.nanquantile(ar_hf, 0.995, axis=1)
del(ar_hf)

with rasterio.open(dir_s+fname+'_large_CC_variable_undestory_db_train.tif') as img_s:
    ar_s = np.float32(img_s.read())
ar_s = np.float64(ar_s[15:, ...])
ar_s[ar_s == 0.] = np.nan
ar_s = ar_s.reshape((ar_s.shape[0], 1, -1)).squeeze()
mu_s = np.nanmean(ar_s, axis=1)
# s_s = np.nanstd(ar_s, axis=1)
min_s = np.nanmin(ar_s, axis=1)
max_s = np.nanmax(ar_s, axis=1)
del(ar_s)

wl_reduced = pd.read_csv('data/dbs/bands.csv')['bands'].values
wl = pd.read_csv('data/Hyperion/wavelength_ids.csv', index_col=0)['wavelength'].values
wl_labels = [f'{d:.2f}'for d in wl_reduced]
bands = np.arange(1, mu_h.shape[0]+1)


# %%
# plotting

x = np.arange(len(wl_reduced))
wl_filtered = np.zeros(wl.shape)
wl_filtered[np.isin(wl, wl_reduced)] = wl[np.isin(wl, wl_reduced)]
wl_filtered[wl_filtered==0] = np.nan

mu_hf_filtered = np.zeros(mu_h.shape) * np.nan
mu_s_filtered = np.zeros(mu_h.shape) * np.nan
min_s_filtered = np.zeros(mu_h.shape) * np.nan
max_s_filtered = np.zeros(mu_h.shape) * np.nan
min_hf_filtered = np.zeros(mu_h.shape) * np.nan
max_hf_filtered = np.zeros(mu_h.shape) * np.nan
j = 0
for i, v in enumerate(wl):
    if v in wl_reduced:
        mu_hf_filtered[i] = mu_hf[j]
        min_hf_filtered[i] = min_hf[j]
        max_hf_filtered[i] = max_hf[j]
        mu_s_filtered[i] = mu_s[j]
        min_s_filtered[i] = min_s[j]
        max_s_filtered[i] = max_s[j]
        j+=1

se =[]
for i,v in enumerate(wl_filtered):
    if i < len(wl_filtered)-1:
        if (np.isnan(v) and not np.isnan(wl_filtered[i+1])):
            se.append(wl_filtered[i+1])
        if (np.isnan(wl_filtered[i+1]) and not np.isnan(v)):
            se.append(v)

fig, ax = plt.subplots(figsize=(FW,0.7*FW))
sns.lineplot(x=wl, y=mu_h, ax=ax, label='Mean unfiltered Hyperion', linestyle=':', color='k')
ax.plot(wl, np.nan*wl, label=' ', visible=False)
ax.plot(wl, mu_hf_filtered, label='Mean filtered Hyperion', linestyle='-',color=palette_diverging[0])
ax.fill_between(wl, max_hf_filtered, min_hf_filtered, color=palette_diverging[0], alpha=0.1, label='Range filtered Hyperion')
ax.plot(wl, mu_s_filtered, label='Mean synthetic', linestyle='-', color=palette_diverging[-2])
ax.fill_between(wl, max_s_filtered, min_s_filtered, color=palette_diverging[-2], alpha=0.2, label='Range synthetic')

ax.xaxis.grid(False)
ax.set_xticks(se)
ax.set_xticklabels([f'{l:.0f}' for l in se], rotation=90)
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('Reflectance')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
fig.tight_layout(rect=(0,0,1,0.95))
# fig.suptitle('Hyperion versus synthetic reflectance')
fig.savefig('./draft/graphics/spectrum_H_and_s.svg')
 
 # %%
