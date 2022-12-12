# %%
# Transform the generated synthetic data
import joblib
import os
import pandas as pd
import rasterio
import numpy as np
from rasterio.merge import merge
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# %%
def csv2tiff(csv_ids, csv_path='data/dbs/', suffix='_large_CC_variable_undestory_db', train_or_test=None, image_width=10, tiff_path=None, target=False, columns=['n']):
    """
    read in csv file, reshape it and save it a s tif
    from tabular csv into image-like tiff 

    Parameters:
    -----------
    csv_dis :  (str, path object or file-like object) 
                csv containing list of ids, or single csv file that is to be transformed
                if a list of filenames is provided, the function expects a header with 
                one column 'fnames' of relevant IDs
                can also be a single ID
    csv_path :  (str, default 'data/dbs/')
                path to input files containing the .csv files that will be transformed
    suffix : (str, default '_large_CC_variable_undestory_db')
                  possible subfix to add in filename after id
    image_width : (int, default 10)
                  width of the image
    tiff_path :  (str, default None) 
                path to directory where the tif files are saved (default: None)
    """
    if isinstance(csv_ids, list):
        fnames = csv_ids
    elif csv_ids.endswith('.csv'):
        df_fnames = pd.read_csv(csv_ids)     
        if 'keep' in df_fnames.columns:
            fnames = df_fnames[df_fnames['keep']==True]['fnames']
        else:
            fnames = df_fnames['fnames']
    else:
        fnames = [csv_ids]
    
    if tiff_path==None:
        tiff_path=csv_path
    
    if target==True:
        suffix = '_'.join(['']+columns)
    
    if train_or_test is not None:
        suffix = suffix+'_'+train_or_test
        
    for fname in fnames:
        df_data = pd.read_csv(csv_path+fname+suffix+'.csv')
        data = df_data.to_numpy().transpose()
        image=data.reshape(data.shape[0], -1, image_width) # band, row, column

        with rasterio.open('data/Hyperion/'+fname+'_Reflectance_topo_v2_Reflectance.img') as src:
            profile = src.profile
            profile.update(
                driver='GTiff',
                width=image.shape[2],
                height=image.shape[1],
                count=image.shape[0],
                dtype =image.dtype
            )
        
        with rasterio.open(tiff_path+fname+suffix+'.tif', 'w', **profile) as dst:
            for l in range(image.shape[0]):
                dst.write(image[l, ...], l+1) 

# csv2tiff('EO1H0430242009236110PT', image_width=int(np.sqrt(10000)))
# %% 
def id_from_path(path, start_sequence='EO1', id_length=22):
    '''Extract image id from file path'''
    _start = path.find(start_sequence)
    return path[_start:_start+id_length]

def make_scaler(path_data_csv, path_scaler, scaler_type='MinMaxScaler', col='signal', start=0):
    '''
    Fit scaler for files from path and save to pkl.
    Assumes the features are along axis 0
    Parameters
    ----------
    path_data_csv : str
        Path to csv that contains all signal and target paths
    path_scaler : str
        path for saving the fitted scaler
    scaler_type : str (default 'MinMaxScaler')
        Any of the scalers available on https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    col : str
        column within path_data_csv; e.g., 'target' or 'signal'
    start : int (default 0)
        first band index to read from, if bands need to be excluded, for example for the synthetic data
    '''
    df = pd.read_csv(path_data_csv)
    for i in range(len(df)):
        path_img = df.iloc[i]['path_signal']
        id = id_from_path(path_img)
        with rasterio.open(df.iloc[i]['path_'+col]) as img:
            array = np.float64(img.read())[start:, ...]

            scaler = getattr(preprocessing, scaler_type)() #MinMaxScaler()
            scaler.fit(array.reshape((-1, array.shape[1]*array.shape[2])).transpose())
            joblib.dump(scaler, path_scaler+id+'_'+col+'_scaler.pkl')

def undo_scaling(scaler, img):
    '''
    Inverse scaling of data using the provided scaler.

    Parameters
    -----------
    scaler : instance of sklearn.preprocessing scaler
        scaler used for unscaling
    img : array-like of shape (n_features, X, Y) or (n_features, Y, X)
        input image
        n_features must match the number of features in scaler
    
    Returns
    -------
    img_unscaled :numpy.ndarray
        image of unscaled values, same shape as input img
    '''
    array = img.reshape((-1, img.shape[1]*img.shape[2])).transpose()
    array = scaler.inverse_transform(array)
    img_unscaled = array.transpose().reshape((-1, img.shape[1], img.shape[2]))
    return img_unscaled


# %%
def create_target_csv(fnames, path_dir='D:/Projects/MA/data/dbs/', suffix='_large_CC_variable_undestory_db', columns=['n'], individual=True, split=True):
    """
    Creates a csv file that contains the path to all signal and target files for the given IDs.

    Parameters:
    -----------
    fnames : (str, path object or file-like object) 
        csv containing list of ids, or single csv file with ids
        if a list of filenames is provided, the function expects a header with 
        one column 'fnames' of relevant IDs
        can also be a single ID
    path_dir : (str, default 'D:/Projects/MA/data/dbs/') 
        path to directory where the csv are found and the tif files saved
    suffix : (str, default '_large_CC_variable_undestory_db')
        suffix added to the filename after id
    """
    if isinstance(fnames, list):
        ids = fnames
    elif fnames.endswith('.csv'):
        df_fnames = pd.read_csv(fnames, index_col=0)     
        if 'keep' in df_fnames.columns:
            ids = df_fnames[df_fnames['keep']==True]['fnames']
        else:
            ids = df_fnames['fnames']
    else:
        ids = [fnames]

    signals = []
    targets = []
    for id in ids:
        df_targets = pd.read_csv(path_dir+id+suffix+'.csv')
        path_target = path_dir+id+'_'.join(['']+columns)
        path_signal = path_dir+id+suffix
        df_targets.to_csv(path_target+'.csv', columns=columns, index=False)
        if split == True:
            signals.append(path_signal+'_train.tif')
            signals.append(path_signal+'_test.tif')
            targets.append(path_target+'_train.tif')
            targets.append(path_target+'_test.tif')
        else:
            signals.append(path_signal+'.tif')
            targets.append(path_target+'.tif')

        if individual:
            if split == True:
                img_paths = {'path_signal': signals[-2:], 'path_target': targets[-2:]}
            else:
                img_paths = {'path_signal': signals[-1:], 'path_target': targets[-1:]}
            df_img = pd.DataFrame(img_paths)
            df_img.to_csv(path_dir+'inform_'+id+'.csv', index=False)

    data = {'path_signal': signals, 'path_target': targets}
    df_data = pd.DataFrame(data)
    df_data.to_csv(path_dir+'inform_reflectance.csv', index=False)    

def train_test_split(fnames, height, width, model_name='inform_',
                    path_dir='D:/Projects/MA/data/dbs/', suffix='_large_CC_variable_undestory_db', columns=['n']):
    """Split data into train and test set; save as csv containing file paths.

    Parameters
    ----------
    fnames : list
        the IDs which are considered
    height : int
        height of the image
    width : int
        width of the image
    model_name : str
        prefix to build full model name
        """
    if isinstance(fnames, list):
        ids = fnames
    elif fnames.endswith('.csv'):
        df_fnames = pd.read_csv(fnames, index_col=0)     
        if 'keep' in df_fnames.columns:
            ids = df_fnames[df_fnames['keep']==True]['fnames']
        else:
            ids = df_fnames['fnames']
    else:
        ids = [fnames]
    for id in ids:
        path_target = path_dir+id+'_'.join(['']+columns)
        path_signal = path_dir+id+suffix
        df_signals = pd.read_csv(path_signal+'.csv')
        df_signals.iloc[0:-(height*width)].to_csv(path_signal+'_train.csv', index=False) #training
        df_signals.iloc[-(height*width):].to_csv(path_signal+'_test.csv', index=False) #testing
        df_targets = pd.read_csv(path_signal+'.csv')
        df_targets.iloc[0:-(height*width)].to_csv(path_target+'_train.csv', columns=columns, index=False) #training
        df_targets.iloc[-(height*width):].to_csv(path_target+'_test.csv', columns=columns, index=False) #testing

        csv_dir = path_dir+model_name+id
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        train_paths = {'path_signal': [path_signal+'_train.tif'], 'path_target': [path_target+'_train.tif']}
        df_train = pd.DataFrame(train_paths)
        df_train.to_csv(csv_dir+'/train.csv', index=False)
        test_paths = {'path_signal': [path_signal+'_test.tif'], 'path_target': [path_target+'_test.tif']}
        df_test = pd.DataFrame(test_paths)
        df_test.to_csv(csv_dir+'/test.csv', index=False)

def make_hyperion_csv(fnames, slices=None, path_dir='D:/Projects/MA/data/Hyperion/', suffix='_Reflectance_topo_v2_Reflectance', individual=False):
    '''
    Create csv of paths for signal and corresponding target.

    Parameters
    ----------
    fnames : str, path object or file-like object
        csv containing list of ids, or single csv file with ids
        if a list of filenames is provided, the function expects a header with 
        one column 'fnames' of relevant IDs
        can also be a single ID
    slices : list (default=None)
        list containing slice names as str
    path_dir : str (default='D:/Projects/MA/data/Hyperion/')
        path to directory containing data (slices)
    suffix : str (default='_Reflectance_topo_v2_Reflectance')
        suffix to append to filename after ID
    individual : bool (default=False)
        if False, save csv containing all (slices of) file ids in fnames
    '''
    if isinstance(fnames, list):
        ids = fnames
    elif fnames.endswith('.csv'):
        df_fnames = pd.read_csv(fnames, index_col=0)     
        if 'keep' in df_fnames.columns:
            ids = df_fnames[df_fnames['keep']==True]['fnames']
        else:
            ids = df_fnames['fnames']
    else:
        ids = [fnames]

    signals_all = []
    for id in ids:  
        # targets.append(path_dir+id+suffix+'_target.tif')
        if slices is not None:
            signals=[path_dir+s+'.tif' for s in slices]
            signals_all + signals
            img_paths = {'path_signal': signals, 'path_target': signals} #targets
            df_img = pd.DataFrame(img_paths)
            df_img.to_csv(path_dir+id+'.csv', index=False)
        else:
            signals_all.append(path_dir+id+suffix+'.tif')
    if not individual:
        data = {'path_signal': signals_all, 'path_target': signals_all} #targets
        df_data = pd.DataFrame(data)
        df_data.to_csv(path_dir+'Hyperion.csv', index=False)

def make_mosaic(path_dir, tag, **kwargs):
    '''
    Make mosaic image from tiles and save to path.

    Parameters
    ----------
    path_dir : str
        path where find the tiles
    tag : str
        tag to identify relevant files
    kwargs : dict
        kwargs passed to rasterio.merge.merge
        if 'dst_path' is not specified, a mosaic.tif is saved in path_dir
    '''
    tiles = []
    for path, subdirs, files in os.walk(path_dir):
        for file in files:
            if file.endswith('.tif') and tag in file:
                tiles.append(rasterio.open(os.path.join(path,file)))
    if 'dst_path' not in kwargs:
        kwargs['dst_path'] = os.path.join(path_dir, 'mosaic.tif')
    print(f'Saving mosaic to {kwargs["dst_path"]}')
    merge(tiles, **kwargs)

def eval(model_name, traits, case='train', dir_path='data/dbs/inform_EO1H0450222004225110PZ/results/'):
    '''
    Calculate evaluation satistics for fit of the model. 

    Parameters
    ----------
    model_name : str
        name of the model e.g., the Hyperion ID
    traits : list
        list of traits (str) that are considered
    case : str (default='train')
        which dataset, e.g., train or test the statistics are computed for and from
    dir_path : str (default='data/dbs/inform_EO1H0450222004225110PZ/results/')    
        path to results for which to calculate the evaluation statistics

    Returns
    -------
    df : pandas.DataFrame
        data frame containing model name, ID, Case, Trait, R2, MAE, and NRMSE
    '''
    if not isinstance(model_name, list):
        model_name = [model_name]
    df = pd.DataFrame(columns=['Model Name', 'ID', 'Case', 'Trait', 'R2', 'MAE', 'NRMSE'])
    for mn in model_name:
        path_results = dir_path+case
        ids = []
        for item in os.scandir(path_results):
            if item.is_dir():
                ids.append(item.name)
        for i, id in enumerate(ids):
            with rasterio.open(os.path.join(path_results,id,'target.tif')) as img:
                target = img.read()
            with rasterio.open(os.path.join(path_results,id,'prediction.tif')) as img:
                prediction = img.read()
            target = target.reshape((target.shape[0], 1, -1)).squeeze()
            prediction = prediction.reshape((prediction.shape[0], 1, -1)).squeeze()
            for t, trait in enumerate(traits):
                r2 = r2_score(target[t, :], prediction[t, :])
                mae = mean_absolute_error(target[t, :], prediction[t, :])
                rmse = np.sqrt(mean_squared_error(target[t, :], prediction[t, :]))
                nrmse = rmse/(np.max(target[t, :]) - np.min(target[t, :]))
                df = pd.concat([df, pd.DataFrame({'Model Name': model_name,
                                                'ID':           id,
                                                'Case':         case,
                                                'Trait':        trait,
                                                'R2':           r2,
                                                'MAE':          mae,
                                                # 'rmse':         rmse,
                                                'NRMSE':        nrmse})], ignore_index=True)
    return df

def filter_array(ar, model_name='EO1H0450222004225110PZ', theta_cc=None, theta_conf=None, theta_lc=None, lct_sp=None):
    '''
    Apply filtering operations equivalent to the filtering for the species analysis.

    Parameters
    ----------
    ar : array-like of shape (n_features, height, width)
        input to which the filtering is applied
    model_name : str (default='EO1H0450222004225110PZ')
        Hyperion ID to identify all relevant files
    theta_cc : float (default=None)
        threshold as minimum value for canopy cover between 0 and 1
        not filtered if None
    theta_conf : float (default=None)
        threshold for minimum confidence in species classification
        not filtered if None
    theta_lc : list (default=None)
        land cover types that remain (int) 210 for conifers, 220 for broadleaves
        not filtered if None
    lct_sp : dict (defaulft=None)
        mapping of land cover type to species numbers
        not filtered if None
    
    Returns
    -------
    ar : numpy.ndarray
        filtered input, clipped to the extent where the reflectance on band 14 is not 0
    '''
    with rasterio.open('./data/secondary_maps/speciesMap/'+model_name+'_species.tif') as img:
        species = np.float32(img.read()).squeeze()
    with rasterio.open('./data/Hyperion/'+model_name+'_Reflectance_topo_v2_Reflectance.img') as img:
        reflectance = img.read(14).squeeze()
    start = np.array([int((reflectance.shape[i]-ar.shape[i+1])/2) for i in [0,1]])
    end = start +1
    end[end==1]=0

    reflectance = reflectance[start[0]:-end[0], start[1]:-end[1]]
    species = species[start[0]:-end[0], start[1]:-end[1]]

    species[reflectance == 0.] = 0.
    ar[:, reflectance == 0.] = 0.
    ar[:, species == 0.] = 0.


    if theta_cc is not None:
        with rasterio.open('./data/secondary_maps/canopyCoverMap/'+model_name+'_canopycover.tif') as img:
            cc = np.float32(img.read()).squeeze()
        cc = cc[start[0]:-end[0], start[1]:-end[1]]
        ar[:, cc<theta_cc] = 0.
    if theta_conf is not None:
        with rasterio.open('./data/secondary_maps/speciesConfidenceMap/'+model_name+'_speciesConfidence.tif') as img:
            sp_conf = np.float32(img.read()).squeeze()
        sp_conf = sp_conf[start[0]:-end[0], start[1]:-end[1]]
        ar[:, sp_conf<theta_conf] = 0.
    if theta_lc is not None:
        with rasterio.open('./data/secondary_maps/landcoverMap/'+model_name+'_landcover.tif') as img:
            lc_type = np.float32(img.read()).squeeze()
        lc_type = lc_type[start[0]:-end[0], start[1]:-end[1]]
        if lct_sp is not None:
            lc_type = np.dstack((lc_type, np.zeros(lc_type.shape)))
            for sp in lct_sp['conifers']:
                lc_type[lc_type[...,0]==sp, 1]=210
            for sp in lct_sp['broadleafes']:
                lc_type[lc_type[...,0]==sp, 1]=220

            lc_type[lc_type[..., 0]!=lc_type[..., 1], 0] = 0
            lc_type = lc_type[..., 0].squeeze()
        for lct in theta_lc:
            lc_type[lc_type == lct] = 999

        ar[:, lc_type==999] = 0.
     
    return ar, species

# -------------------------------------------------------------------------------------------------------------

def stretch(ar):
    ar=ar.astype(float)
    ar[ar<=0]=np.nan
    m=np.nanmean(ar)
    std=np.nanstd(ar)
    ar[ar<m-3*std]=m-3*std
    ar[ar>m+3*std]=m+3*std
    ar-=np.nanmin(ar)
    ar=ar/np.nanmax(ar)
    return ar

def RGB_matrix(R,G,B):
    B=stretch(B)
    G=stretch(G)
    R=stretch(R)
    mat=np.dstack((B,G,R))
    return mat

#def get_understory_spectra(hyper,lc,understory=[40,50,100]):
#    for c in np.unique(lc):
#        if c not in understory:
#            lc[lc==c]=0
#    hyper[lc==0,:]=0
#    return np.nanmean(hyper[hyper[:,:,10]>0,:],axis=0)

def get_understory_spectra(hyper,lc,understory=[40,50,100]):
    for c in np.unique(lc):
        if c not in understory:
            lc[lc==c]=0
    hyper[lc==0,:]=0
    
    return np.nanmean(hyper[hyper[:,:,10]>0,:],axis=0), np.nanmedian(hyper[hyper[:,:,10]>0,:],axis=0), np.nanpercentile(hyper[hyper[:,:,10]>0,:],10,axis=0), np.nanpercentile(hyper[hyper[:,:,10]>0,:],90,axis=0),hyper[hyper[:,:,10]>0,:]
    
    # ndvi=(hyper[:,:,81]-hyper[:,:,39])/(hyper[:,:,81]+hyper[:,:,39])
    # return np.nanmean(hyper[np.where((ndvi<100) & (0<ndvi))],axis=0), np.nanmedian(hyper[np.where((ndvi<100) & (0<ndvi))],axis=0), np.nanpercentile(hyper[np.where((ndvi<100) & (0<ndvi))],10,axis=0), np.nanpercentile(hyper[np.where((ndvi<100) & (0<ndvi))],90,axis=0), hyper[np.where((ndvi<100) & (0<ndvi))]

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

def add_SNR(spectra,bands):
    spectra=spectra[:,bands>400]
    bands=bands[bands>400]
    spectra=spectra[:,bands<2365]
    bands=bands[bands<2365]
    SNRb=[  400,550,650,700,940,1025,1110,1120,1225,1326,1336,1498,1518,1575,1770,1780,1900,2000,2125,2300,2365]
    SNRval=[  0,161,144,147, 25,  90,  90,   5, 110, 110,   0,   0,  89,  89,  89,   0,   0,  40,  40,  25,   0]
    f=interp1d(SNRb,SNRval)
    SNR=f(bands)
    NSR=1/SNR
    rand=np.random.normal(loc=0,scale=NSR,size=spectra.shape)
    spectra=spectra+rand*NSR*spectra
    return spectra, bands

def apply_hyperion_snr(array,bands):
        array=array[...,bands<2365]
        bands=bands[bands<2365]
        #getting the SNR@50%
        SNRb=[  400,550,650,700,940,1025,1110,1120,1225,1326,1336,1498,1518,1575,1770,1780,1900,2000,2125,2300,2365]
        SNRval=[  0,161,144,147, 25,  90,  90,   5, 110, 110,   0,   0,  89,  89,  89,   0,   0,  40,  40,  25,   0]
        SNRval = [x+1 for x in SNRval]
        f=interp1d(SNRb,SNRval)
        SNR=f(bands)
        NSR=0.5/(SNR)
        rand=np.random.normal(loc=0,scale=NSR,size=array.shape)
        #fig,ax=plt.subplots()
        #ax.plot(bands,rand[0,:])
        #rand=np.random.normal(loc=0,scale=0.01,size=bands.shape)
        #ax.plot(bands,rand)
        #plt.show()
        array+=rand
        return array,bands

def preproc_hyp(X,bands):

    X[X>1] = 1
    X[...,np.logical_and(bands>=450,bands<=740)] = savgol_filter(X[...,np.logical_and(bands>=450,bands<=740)],5,2)
    X[...,np.logical_and(bands>=790,bands<=890)] = savgol_filter(X[...,np.logical_and(bands>=790,bands<=890)],5,2)
    X[...,np.logical_and(bands>=990,bands<=1080)] = savgol_filter(X[...,np.logical_and(bands>=990,bands<=1080)],5,2)
    X[...,np.logical_and(bands>=1185,bands<=1310)] = savgol_filter(X[...,np.logical_and(bands>=1185,bands<=1310)],5,2)
    X[...,np.logical_and(bands>=1510,bands<=1750)] = savgol_filter(X[...,np.logical_and(bands>=1510,bands<=1750)],5,2)
    X[...,np.logical_and(bands>=2060,bands<=2340)] = savgol_filter(X[...,np.logical_and(bands>=2060,bands<=2340)],5,2)
    return X,bands

def remove_bands(array,bands):
    """
    omit all bands
    <450 (500), in between 751 - 789, 890 - 980, 1104 - 1510, 1310 - 1509, 1750 - 2060, or >=2340
    """
    
    try:
        array = array[...,bands>=500] #500
        bands = bands[bands>=500] #500
    except:
        pass
    try:
        array = array[...,np.logical_or(bands<=740,bands>=790)]
        bands = bands[np.logical_or(bands<=740,bands>=790)]
    except:
        pass
    try:
        array = array[...,np.logical_or(bands<890,bands>990)]
        bands = bands[np.logical_or(bands<890,bands>990)]
    except:
        pass
    try:
        array = array[...,np.logical_or(bands<1080,bands>1185)]
        bands = bands[np.logical_or(bands<1080,bands>1185)]
    except:
        pass
    try:
        array = array[...,np.logical_or(bands<1310,bands>=1510)]
        bands = bands[np.logical_or(bands<1310,bands>=1510)]
    except:
        pass
    try:
        array = array[...,np.logical_or(bands<1750,bands>2060)]
        bands = bands[np.logical_or(bands<1750,bands>2060)]
    except:
        pass
    try:
        array = array[...,bands<2250]
        bands = bands[bands<2250]
    except:
        pass
    return array,bands

