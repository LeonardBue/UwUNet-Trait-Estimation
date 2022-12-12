# %% generate inform inputs
import rasterio as rio
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyDOE2
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import sys
from tqdm import tqdm

sys.path.append('./code/')
import utils
import inform

if __name__=='__main__':
    """
    exposed/barren land -> 33
    bryoids             -> 40
    shrubs              -> 50
    herbs               -> 100
    """
    
    dir_r = 'data/Hyperion/'#'D:/Hyperion_images/CABO_reflectance/'
    dir_lc = 'data/secondary_maps/landcoverMap/' #'E:/Hyperion_images/landcoverMap/'
    image_ids = 'data/Hyperion/images_visual_check.csv'
    df_toKeep = pd.read_csv(image_ids)
    fnames_toKeep = df_toKeep['fnames'].values
    keeps = df_toKeep['keep'].values
    fnames = []
    for fname in fnames_toKeep:
        if keeps[fnames_toKeep==fname]==True:
            fnames.append(fname)
    
    fnames= ['EO1H0450222004225110PZ']
    cols =  ['cab','prot','cbc','lai']# ['cab','car','ewt','prot','cbc','lma','lai', 'cc'] # target values

    im_width =  512
    im_height = 1024+128
    nsamples = im_height * im_width

    im_width_test = 512 # no testing if one is set to 0
    im_height_test = 128 #128
    # %%
    # ------------------------------------------------------------------------

    #get list of quebec images
    # df_list_img=pd.read_csv('C:/Users/tmiragli/CABO/apply_PLSR_hyperion/quebec_specific/quebec_metadata.csv',encoding='unicode_escape')
    # fnames_img_quebec=df_list_img['Entity ID'].str[:-7].tolist()
    # fnames=list(set(fnames).intersection(set(fnames_img_quebec)))
    
    for fname in fnames:
        print(fname)
        prefix_lc = '_landcover.tif'
        try:
            prefix_r='_Reflectance_topo_v2_Reflectance.img'
            #open hyperspectral and landcover images
            img=envi.open(dir_r+fname+prefix_r[:-4]+'.hdr')
            bands=np.asarray(img.metadata['wavelength']).astype(float)
            bands_original=bands.copy()
            fwhm=np.asarray(img.metadata['fwhm']).astype(float)
            tts=float(img.metadata['sun zenith'])
            sunAzim=float(img.metadata['sun azimuth'])
            satelliteAzim=float(img.metadata['satellite azimuth'])
            phi=sunAzim-satelliteAzim
            if phi<0:
                phi=phi+360
            tto=np.abs(float(img.metadata['satellite zenith']))
            img = rio.open(dir_r+fname+prefix_r)
            ar = img.read()
            ar = np.moveaxis(ar,0,2)
        except:
            print(f'could not generate inform data for id: {fname}')

        img = rio.open(dir_lc+fname+prefix_lc)
        lc = img.read(1)
        #get soil reflectance
        understory_spectra,_,_,_,all_spectra = utils.get_understory_spectra(ar,lc)
        all_spectra = all_spectra/1e4
        understory_spectra = understory_spectra/1e4
        all_spectra, bands = utils.remove_bands(all_spectra, bands_original)
        all_spectra,_ = utils.preproc_hyp(all_spectra,bands)
        print(understory_spectra.shape)
        understory_spectra, bands = utils.remove_bands(understory_spectra, bands_original)
        print(understory_spectra.shape)
        understory_spectra,_ = utils.preproc_hyp(understory_spectra,bands)
        f = interp1d(bands,understory_spectra,bounds_error=False,fill_value=0)
        understory_spectra = f(np.arange(400,2501))
        fall = interp1d(bands,all_spectra,bounds_error=False,fill_value=0)
        all_spectra = fall(np.arange(400,2501)).astype(np.float16)
        all_spectra.dump('./data/dbs/'+fname+'_understory_spectra.npy')

        use_mean = True
        mean_spectra = np.nanmean(all_spectra, axis=0)
        if use_mean:
            all_spectra = mean_spectra.reshape(1, len(mean_spectra))
        
        #run LHS
        tto=np.round(tto,0)
        tts=np.round(tts,0)
        phi=np.round(phi,0)
        N = [1.0,3.5]
        # LMA = [0.002,0.025]
        PROT = [0.0002,0.003]
        CBC = [0.002,0.03]
        CAR = [8,30]
        CAB = [15,120]
        EWT = [0.002,0.03]
        cbrown = 0
        LAI = [1,10]
        LIDFA = [30,70]
        lidfb = 0
        rsoil = 1
        psoil = 1
        hspot = 0.01
        typelidf = 2
        SD = [50,1500]
        CD = [2,8]
        H = [4,17]
        CC = [0.6, 1]
        lhs = pyDOE2.lhs(12,samples=nsamples,criterion='center', random_state=42)
        k = 0
        params = np.zeros((nsamples,14))
        values = np.zeros((nsamples,2101))
        for entry in tqdm(lhs):
            n = entry[0]*(N[1]-N[0])+N[0]
            prot = entry[1]*(PROT[1]-PROT[0])+PROT[0]
            cbc = entry[2]*(CBC[1]-CBC[0])+CBC[0]
            lma = prot+cbc
            lai = entry[3]*(LAI[1]-LAI[0])+LAI[0]
            lidfa = entry[4]*(LIDFA[1]-LIDFA[0])+LIDFA[0]
            car = entry[5]*(CAR[1]-CAR[0])+CAR[0]
            sd = entry[6]*(SD[1]-SD[0])+SD[0]
            cc_min = np.pi * CD[0]**2 /4 * sd/1e4
            cc_max = np.pi * CD[1]**2 /4 * sd/1e4
            cc = entry[7]*(np.amin([CC[1],cc_max])-np.amax([CC[0],cc_min]))+np.amax([CC[0],cc_min])
            # cd_min = 2*np.sqrt(CC[0]*1e4/(np.pi*sd))
            # cd_max = 2*np.sqrt(CC[1]*1e4/(np.pi*sd))
            # cd = entry[7]*(np.amin([CD[1],cd_max])-np.amax([CD[0],cd_min]))+np.amax([CD[0],cd_min])
            h = entry[8]*(H[1]-H[0])+H[0]
            cab = entry[9]*(CAB[1]-CAB[0])+CAB[0]
            ewt = entry[10]*(EWT[1]-EWT[0])+EWT[0]
            idx_understory = (entry[11]*(all_spectra.shape[0]-1)).astype(int)
            # cc = np.pi * cd**2 /4 * sd/1e4 # estimate canopy cover 
            cd = 2*np.sqrt(cc*1e4/(np.pi*sd)) # estimate crown density
            lai_s = lai * cc # calculate LAI_scene from LAI_tree using the canopy cover

            params[k,:] = [n,cab,car,ewt,prot,cbc,lma,lai_s,lidfa,sd,cd,h,idx_understory,cc]
            if k==0:
                print('ASSESS')
                print(tto)
                print(tts)
                rho_canopy = inform.run_inform(n, cab, car, cbrown, ewt, 0, lai, tts, tto, phi, sd, cd, h, typelidf=2, lidfb=0, lidfa=lidfa, hspot=hspot,rsoil=rsoil,psoil=psoil, ant=0, prot=prot, cbc=cbc, alpha=40.0, prospect_version='PRO', rsoil0=all_spectra[idx_understory,:],model_understory_reflectance=False)

                if np.isnan(np.sum(rho_canopy)):
                    TTS = np.arange(-3,3.1,1)
                    TTO = np.arange(-3,3.1,1)
                    fs = 0
                    fo = 0
                    d = 100
                    for s in TTS:
                        for o in TTO:
                            rho_canopy = inform.run_inform(n, cab, car, cbrown, ewt, 0, lai, tts+s, tto+o, phi, sd, cd, h, typelidf=2, lidfb=0, lidfa=lidfa, hspot=hspot,rsoil=rsoil,psoil=psoil, ant=0, prot=prot,cbc=cbc,  alpha=40.0, prospect_version='PRO', rsoil0=all_spectra[idx_understory,:],model_understory_reflectance=False)
                            if not np.isnan(np.sum(rho_canopy)):
                                if np.sqrt(s**2+o**2)<d:
                                    fs = s
                                    fo = o
                                    d = np.sqrt(s**2+o**2)
                    tto = tto+fo
                    tts = tts+fs
                print(tto)
                print(tts)
                rho_canopy = inform.run_inform(n, cab, car, cbrown, ewt, 0, lai, tts, tto, phi, sd, cd, h, typelidf=2, lidfb=0, lidfa=lidfa, hspot=hspot,rsoil=rsoil,psoil=psoil, ant=0, prot=prot,cbc=cbc,  alpha=40.0, prospect_version='PRO', rsoil0=all_spectra[idx_understory,:],model_understory_reflectance=False)
            rho_canopy = inform.run_inform(n, cab, car, cbrown, ewt, 0, lai, tts, tto, phi, sd, cd, h, typelidf=2, lidfb=0, lidfa=lidfa, hspot=hspot,rsoil=rsoil,psoil=psoil, ant=0, prot=prot,cbc=cbc,  alpha=40.0, prospect_version='PRO', rsoil0=all_spectra[idx_understory,:],model_understory_reflectance=False)
            values[k,:] = rho_canopy
            k += 1

        fig,ax=plt.subplots()
        ax.plot(np.arange(400,2501),values[0,:])    
        print(values[0,:])

        #degrading to match Hyperion (FWHM) and interpolate
        spectra = gaussian_filter(values,sigma=(0,utils.fwhm2sigma(np.mean(fwhm))))
        f = interp1d(np.arange(400,2501),spectra)
        spectra = f(bands_original)
        spectra, bands_reduced = utils.remove_bands(spectra, bands_original)
        spectra, bands_reduced = utils.preproc_hyp(spectra, bands_reduced)
        #add noise
        spectra, bands_noisy = utils.apply_hyperion_snr(spectra, bands_reduced)

        pd.DataFrame({'bands':bands_noisy}).to_csv('./data/dbs/bands.csv', index=False)

        #create dataframe
        data = {}
        k = 0
        for elem in ['n','cab','car','ewt','prot','cbc','lma','lai','lidfa','sd','cd','h','understory_index','cc']:
            data[elem] = params[:,k]
            k += 1
        k = 0
        for elem in bands_noisy:
            data[str(elem)] = spectra[:,k]
            k += 1
        
        df = pd.DataFrame.from_dict(data)

        print(df)
        df.to_csv('./data/dbs/'+fname+'_large_CC_variable_undestory_db.csv')

        # calculate mean reflectance from generated soil reflectances
        rsoil_mean = df.iloc[:, df.columns.get_loc('cc')+1:].mean(axis=0)
        df_rsoil = pd.DataFrame(rsoil_mean)
        df_rsoil.to_csv('./data/dbs/'+fname+'_large_CC_variable_undestory_db_rsoil.csv')
# %%
# split in train and test
if ((im_width_test != 0) and (im_height_test != 0)):
    utils.train_test_split(fnames, im_height_test, im_width_test, model_name='inform_',
                        path_dir='D:/Projects/MA/data/dbs/', suffix='_large_CC_variable_undestory_db', columns=cols)

# Transform inputs
try:
    utils.csv2tiff(fnames, train_or_test='train', image_width=im_width, columns=cols)
    utils.csv2tiff(fnames, train_or_test='test', image_width=im_width_test, columns=cols)
    print('signal converted to tiff')

except Exception as E:
    print('could not transform signals to tiff :/')
    print(E)

# Tranbsform targets
id_path = 'data/Hyperion/images_visual_check_only_1.csv'
try:
    # for multiple images substitute fnames[0] with id_path
    utils.create_target_csv(id_path, path_dir='D:/Projects/MA/data/dbs/', suffix='_large_CC_variable_undestory_db', columns=cols, split=True)
    utils.csv2tiff(fnames, train_or_test='train', image_width=im_width, target=True, columns=cols)
    utils.csv2tiff(fnames, train_or_test='test', image_width=im_width_test, target=True, columns=cols)
    print('target converted to tiff')
except Exception as E:
    print('could not transform targets to tiff :/')
    print(E)

# %% plot some spectra for other files just read in from csv
# s_rsoil = df.iloc[:, df.columns.get_loc("understory_index")+1:].std(axis=0)
# min_rsoil = df.iloc[:, df.columns.get_loc("understory_index")+1:].min(axis=0)
# max_rsoil = df.iloc[:, df.columns.get_loc("understory_index")+1:].max(axis=0)
# fig = plt.figure(figsize=(10,10))
# plt.fill_between(range(len(rsoil_mean)), max_rsoil, min_rsoil, alpha=0.3)
# plt.fill_between(range(len(rsoil_mean)), rsoil_mean+s_rsoil, rsoil_mean-s_rsoil, alpha=0.4)
# plt.plot(range(len(rsoil_mean)), rsoil_mean)
# plt.grid()
# fig.legend()
