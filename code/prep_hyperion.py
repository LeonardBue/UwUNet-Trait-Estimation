# generate inform inputs
import rasterio as rio
import spectral.io.envi as envi
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from itertools import repeat, cycle
from rasterio.windows import Window, WindowMethodsMixin
from tqdm import tqdm

sys.path.append('./code/')
import utils

def _calc_tile_size(size, ts, ps):
    '''Recursive calculation of tile size to minimize cut-off'''
    if ((size % ts) > (ps)) : #and ts > 2*ps:
        return int(_calc_tile_size(size, ts/2, ps))
    else:
        return int(ts)

def get_tile_size(width, height, patch_size, max_pixels=512*512):
    '''Calculate tile sizes in x and y dimension based on patch_size and max pixels in image with shape zyx'''
    ts_x = width//patch_size * patch_size
    ts_y_max = 2**int(np.floor(math.log(max_pixels/ts_x, 2)))
    n = 1
    while ts_y_max < patch_size:
        n +=1
        ts_x = int((width/n)//patch_size * patch_size)
        ts_y_max = 2**int(np.floor(math.log(max_pixels/ts_x, 2)))
    ts_y = _calc_tile_size(height, ts_y_max, patch_size)
    return [ts_x, ts_y]

def get_slices(width, height, ts, patch_size):
    '''calculte starting incdices for tiles in x and y direction'''
    offset = [int(width%patch_size/2), int(height%patch_size/2)]

    starts_x = np.array([(offset[0]+ts[0]*i) for i in range(width//ts[0])])
    starts_y = np.array([(offset[1]+ts[1]*i) for i in range(height//ts[1])])
    return starts_x, starts_y

def crop_image(ar, ts):
    '''crop image left and right to be multiple of patch size and same for height'''
    print(ar.shape)
    ar = ar[:, int((ar.shape[1]%ts[0])/2):-int((ar.shape[1]%ts[0])/2+1),:]
    ar = ar[..., int((ar.shape[2]-ts[1])/2):-int((ar.shape[2]-ts[1])/2+1)]
    print(ar.shape)
    return ar

def slice_image(ar, ts):
    '''create tiles that are no more pixels than 1024*2048, have the same size and are dividable by patch_size'''
    ar_shape = ar.shape
    cropped_imgs = []
    for i in range(int(math.ceil(ar_shape[1]/(ts[0] * 1.0)))):
        for j in range(int(math.ceil(ar_shape[2]/(ts[1] * 1.0)))):
            tile = ar[:, ts[0]*i:min(ts[0]*i+ts[0], ar_shape[1]), ts[1]*j:min(ts[1]*j+ts[1], ar_shape[2])]
            cropped_imgs.append(tile)
    return cropped_imgs

def crop_and_slice(id, patch_size, dir_path='data/Hyperion/', suffix='_Reflectance_topo_v2_Reflectance.img'):
    with rio.open(dir_path+id+suffix) as img:
        ar = img.read() #zyx
        profile = img.profile
    ts = get_tile_size(ar.shape[1], ar.shape[2], patch_size)
    ar = crop_image(ar, ts)
    tiles = slice_image(ar, ts)
        
    for i, tile in enumerate(tiles):
        with rio.open(dir_path+id+'_cropped_'+i+'.tif', 'w', **profile) as dst:
            profile.update(
                driver='GTiff',
                width=ts[1],
                height=ts[0],
                count=tile.shape[0],
                dtype=tile.dtype
            )
            for l in range(tile.shape[0]):
                dst.write(tile[l, ...], l+1)

def main():
    patch_size = 64
    max_pixels = 64*128
    dir_r = 'data/Hyperion/'
    image_ids = 'data/Hyperion/images_visual_check.csv'
    # df_toKeep = pd.read_csv(image_ids)
    # fnames_toKeep = df_toKeep['fnames'].values
    # keeps = df_toKeep['keep'].values
    # fnames = []
    # for fname in fnames_toKeep:  
    #     if keeps[fnames_toKeep==fname]==True:
    #         fnames.append(fname)
    
    fnames= ['EO1H0450222004225110PZ']

# ------------------------------------------------------------------------
    slices = []
    for fname in tqdm(fnames, colour='blue'):
        print(fname)
        path_cropped = dir_r+'cropped/'
        if not os.path.exists(path_cropped):
            os.makedirs(path_cropped)
        try:
            suffix='_Reflectance_topo_v2_Reflectance.img'
            #open hyperspectral and landcover images
            img=envi.open(dir_r+fname+suffix[:-4]+'.hdr')
            bands=np.asarray(img.metadata['wavelength']).astype(float)
        except:
            suffix = '_Reflectance_flat_v2_Reflectance.img'
            #open hyperspectral and landcover images
            img = envi.open(dir_r+fname+suffix[:-4]+'.hdr')
            bands = np.asarray(img.metadata['wavelength']).astype(float)
        
        with rio.open('data/Hyperion/'+fname+'_Reflectance_topo_v2_Reflectance.img') as src:
            profile = src.profile
            ts = get_tile_size(src.width, src.height, patch_size, max_pixels)
            starts_x, starts_y = get_slices(src.width, src.height, ts, patch_size)
            start_grid = np.meshgrid(starts_x, starts_y)
            print(ts)
            
            i=0
            for offset in tqdm(zip(start_grid[0].reshape((-1,1)).squeeze(), start_grid[1].reshape((-1,1)).squeeze()), total=len(starts_x)*len(starts_y), colour = 'green'):
                window = Window(*offset, *ts)
                ar = src.read(window=window)
                ar = ar/1e4
                ar = np.moveaxis(ar, 0, 2) #yxz
                
                if not (ar == 0).all():
                    win_transform = src.window_transform(window)
                    spectra, bands_reduced = utils.remove_bands(ar, bands)
                    spectra, bands_reduced = utils.preproc_hyp(spectra, bands_reduced)
                    spectra[spectra<0] = 0.
                    spectra[spectra>1] = 1.

                    # band, row, column
                    spectra = np.moveaxis(spectra, 2, 0) #zyx
                    profile.update(
                        driver='GTiff',
                        width=window.width,
                        height=window.height,
                        count=spectra.shape[0],
                        dtype=spectra.dtype,
                        transform=win_transform
                    )
                    slices.append(f'{fname}_{i+1}')

                    with rio.open(path_cropped+fname+f'_{i+1}.tif', 'w', **profile) as dst:
                        for l in range(spectra.shape[0]):
                            dst.write(spectra[l, ...], l+1)
                    i +=1

        utils.make_hyperion_csv(fname, slices[-(i+1):], path_dir='D:/Projects/MA/'+path_cropped, suffix='', individual=True)
    utils.make_hyperion_csv(slices, suffix='')


if __name__ == '__main__':
    main()
    