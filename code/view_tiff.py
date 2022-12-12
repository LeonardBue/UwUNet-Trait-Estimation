# %%
import sys
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
import numpy as np

def view(file, source='Hyperion', scale_factor=1e3, band=None, style='rgb'):
    """
    View tiff files like from Hyperion

    To view a file from the command line run the command in the following way:

    $ python view_tiff.py <FILENAME> source=Hyperion style=rgb

    to find a wavelength 'w', run the following code and add +1 to the index:
    df = pd.read_csv('D:\Projects\MA\data\Hyperion\wavelength_ids.csv', index_col=0)
    df.iloc[(df['wavelength'] - w).abs().argsort()[0],:]
    """
    cmap='viridis'
    with rio.open(file) as img:
            # shifted by number of additional input bands (14)
        if band is not None:
            arr = np.float64(img.read(int(band)+1))
            print(f'min: {np.nanmin(arr)} \t max: {np.nanmax(arr)}')
            show(arr, cmap=cmap)
        else:
            if source=='synthetic':
                shift = 14
                scale_factor=1/10
                vmax=None
                arr = np.float64(img.read())
            else:
                shift = 0
                vmax=50
                arr = np.float64(img.read())
                arr[arr > 10000.] = np.nan

            if style=='ndvi':
                bands = np.array([23, 40]) # R650 -> 23, NIR825 -> 40, NIR925 -> 50, SWIR1245 -> 82, SWIR1648 -> 122
                print(f'Showing NDVI calculated from bands {bands[0]} and {bands[1]}; source: {source}')
                R = arr[bands[0].item()-1+shift, ...]
                NIR = arr[bands[1].item()-1+shift, ...]
                NDVI = (NIR - R)/(NIR + R)
                NDVI[NDVI>1] = np.nan
                show(NDVI, cmap=cmap, vmax=vmax)
            elif style=='n':
                N = arr[0, ...]
                show(N, cmap=cmap)
            elif style=='lma':
                lma = arr[1, ...]
                show(lma, cmap=cmap)

            else:
                match style:
                    case 'rgb':
                        bands = np.array([23, 14, 4]) # bands corresponding to wavelengths of 650,  560, and 457 nm
                        print(f'Showing RGB representation with bands {bands}; source: {source}')
                    case 'fc':
                        bands = np.array([50, 14, 4]) # replace red with NIR
                        print(f'Showing false color representation with bands {bands}; source: {source}')
                bands += shift
                bands
                RGB = np.stack([arr[bands[0].item()-1, ...], 
                                arr[bands[1].item()-1, ...], 
                                arr[bands[2].item()-1, ...]], axis=2).transpose(2, 0, 1)
                show(RGB/scale_factor)


# view('D:\Projects\MA\data\Hyperion\EO1H0430242009236110PT_Reflectance_topo_v2_Reflectance.img', source='Hyperion', style='ndvi')
# %%    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        view(sys.argv[1], **dict(arg.split('=') for arg in sys.argv[2:]))
    else:
        view(sys.argv[1])
