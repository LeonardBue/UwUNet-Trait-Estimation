# %%
# define plotting functions
from turtle import color
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
import pandas as pd
import rasterio
import seaborn as sns
import statsmodels.stats.weightstats as ws
import sys
sys.path.append('./code')

from IPython.display import display
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import ttest_ind, shapiro
from skimage.metrics import structural_similarity as ssim
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from utils import make_mosaic, eval, filter_array

FW = 5.7885 # text width in Latex

def get_results(dir_path, ids, n_outputs=2, combined=True):
    if not isinstance(ids, list):
        with rasterio.open(dir_path+ids+'/target.tif') as target_img:
            target = np.float64(target_img.read())
        with rasterio.open(dir_path+ids+'/prediction.tif') as prediction_img:
            prediction = np.float64(prediction_img.read())
        return target, prediction
    else:
        targets = {}
        preds = {}
        for i, id in enumerate(ids):
            target_img = rasterio.open(dir_path+id+'/target.tif')
            target = np.float64(target_img.read())
            prediction_img = rasterio.open(dir_path+id+'/prediction.tif')
            prediction = np.float64(prediction_img.read())
            if combined==False:
                targets[id] = target.reshape((n_outputs, 1, -1)).squeeze()
                preds[id] = prediction.reshape((n_outputs, 1, -1)).squeeze()
            else:
                if i==0:
                    targets = target.reshape((n_outputs, 1, -1)).squeeze() 
                    preds = prediction.reshape((n_outputs, 1, -1)).squeeze()
                else:
                    targets = np.concatenate((targets, target.reshape((n_outputs, 1, -1)).squeeze()), axis=0)
                    preds = np.concatenate((preds, prediction.reshape((n_outputs, 1, -1)).squeeze()), axis=0)

        return targets, preds

def learning_curve(dir_path, title, ylim=[0,0.1], yscale=None, f_path='./draft/graphics/learning_curve.svg'):
    tr_loss = pd.read_csv(dir_path+'/losses.csv')
    
    fig, ax = plt.subplots(1,1, figsize=(FW,0.7*FW))
    sns.lineplot(ax=ax, x=tr_loss.num_iter, y=tr_loss.loss_batch, label='Training', color=palette_diverging[0])# color=cp_train[int(len(cp_train)/2)])
    # ax.plot(tr_loss.num_iter, tr_loss.loss_batch, label='Training loss')
    try:
        val_loss = pd.read_csv(dir_path+'/losses_val.csv')
        sns.lineplot(ax=ax, x=val_loss.num_iter, y=val_loss.loss_val, label='Validation', color=palette_diverging[-2])
        # ax.plot(val_loss.num_iter, val_loss.loss_val, label='Validation loss')
    except Exception as E:
        print(f'No validation data found:\n{E}')
    
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Mean Squared Error')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(visible=True, which='minor', linewidth=0.5, axis='y')
    ax.legend()
    # fig.suptitle(title)
    fig.tight_layout(rect=(0,0,1,0.95))
    fig.savefig(f_path)

def scatter_predictions(dir_path, ids, trait='N', band=0, n_outputs=1, f_path='./draft/graphics/scatter_'):
    
    if not isinstance(dir_path, list):
        dir_path = [dir_path]
    
    fig, ax = plt.subplots(1, 2, figsize=(2*FW,2*0.33*FW), sharey=True)
    for i in range(len(dir_path)):
        targets, preds = get_results(dir_path[i], ids[i], n_outputs)
        if targets.ndim > 1:
            targets = targets[band]
        if preds.ndim > 1:
            preds = preds[band]
        if 'train' in dir_path[i]:
            title = 'Training'
        else:
            title = 'Validation'
        
        # Create linear regression object
        regr = linear_model.LinearRegression()
        regr.fit(targets.reshape(-1, 1), preds.reshape(-1, 1))
        x = np.array([np.nanmin(targets), np.nanmax(targets)]).reshape(-1, 1)
        y = regr.predict(x)
        # sns.scatterplot(ax=ax[i], x=targets, y=preds, marker='.', alpha=0.05, edgecolor='none', color=palette[4])
        hb = ax[i].hexbin(targets, preds, gridsize=50, mincnt=1, cmap='Blues') #cmap.reversed())
        sns.lineplot(ax=ax[i], x=x.squeeze(), y=x.squeeze(), color='k', linestyle='--', label='Target', legend=False)
        sns.lineplot(ax=ax[i], x=x.squeeze(), y=y.squeeze(), color='k', label='Linear regression', legend=False)#palette[int(len(palette)/2)])
  
        cb=fig.colorbar(hb, ax=ax[i])
        cb.set_label('Counts')

        ax[i].set_xlabel(f'Target {trait.upper()} {get_units([trait])[0]}')
        # ax[i].set_title(title)
        # ax[i].grid(False)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    # fig.suptitle(f'{trait} Estimates over Target Values')
    ax[0].set_ylabel(f'Estimated {trait.upper()} {get_units([trait])[0]}')
    fig.tight_layout(rect=(0,0,1,0.95))
    # fig.show()
    fig.savefig(f_path+trait+'.png', dpi=600)

def optuna_analyse_p(path_study):
    study = joblib.load(path_study)
    df = study.trials_dataframe()
    df.sort_values('value')
    best10 = np.empty(shape=(6, 10, 3))
    for i, p in enumerate(np.arange(0.0, 0.6, 0.1)):
        best10[i, :, 0] = df[df.params_p == p].iloc[:10]['params_p']
        best10[i, :, 1] = df[df.params_p == p].iloc[:10]['params_lr']
        best10[i, :, 2] = df[df.params_p == p].iloc[:10]['value']
    best=best10.reshape((1, -1, 3)).squeeze()

    fig, ax = plt.subplots()
    sns.scatterplot(ax=ax, x = best[:,0], y=best[:,1], hue=best[:,2], palette=cmap)
    ax.set_yscale('log')

    norm = plt.Normalize(best[:,2].min(), best[:,2].max())
    sm = plt.cm.ScalarMappable(cmap=ListedColormap(cmap.colors[::-1]), norm=norm)
    sm.set_array([])
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    cbar = ax.figure.colorbar(sm, cax=cax)
    ax.grid(b=True, which='minor', linewidth=0.5)
    ax.get_legend().remove()
    ax.set_xlabel(r'Dropout rate $p$')
    ax.set_ylabel(r'Learning rate $lr$')
    # fig.suptitle('10 Best Trial for each Dropout Rate')
    cbar.set_label('MSE after 10 iterations')
    fig.savefig('./draft/graphics/optuna_scatter.png', dpi=600)

def compare_images(dir_path, ids, trait='N', n_outputs=2, individual=False):
    match trait:
        case 'N':
            band = 0
        case 'LMA':
            band = 1
        case 'Cab':
            band = 2
    if n_outputs==1:
        band=0
    if not isinstance(dir_path, list):
        dir_path = [dir_path]
    
    # if individual:
    #     model_name=dir_path.copy()
    #     dir_path=[dir_path]*2

    # Create linear regression object
    regr = linear_model.LinearRegression()

    fig, ax = plt.subplots(1, 2, figsize=(FW,0.5*FW))
    for i in range(len(dir_path)):
        targets, preds = get_results(dir_path[i], ids[i], n_outputs, combined=False)
        if 'train' in dir_path[i]:
            title = 'Training'
            cp = cp_train
        else:
            title = 'Validation'
            cp = cp_test
        for j, id in enumerate(ids[i]):
            if targets[id].ndim > 1:
                target_values = targets[id][band]
            if preds[id].ndim > 1:
                pred_values = preds[id][band]
            # Train the model using the training sets
            regr.fit(target_values.reshape(-1, 1), pred_values.reshape(-1, 1))
            x = np.array([np.nanmin(target_values), np.nanmax(target_values)]).reshape(-1, 1)
            y = regr.predict(x)
            sns.lineplot(ax=ax[i], x=x.squeeze(), y=y.squeeze(), color=cp[j])
        sns.lineplot(ax=ax[i], x=x.squeeze(), y=x.squeeze(), color='k')
        ax[i].set_xlabel(f'Target {trait.upper()}')

        # ax[i].set_title(title)
        # ax[i].get_legend().remove()
    # fig.suptitle(f'{trait} Estimates over Target Values')
    fig.supylabel(f'Estimated {trait.upper()}')
    fig.tight_layout(pad=2)
    fig.savefig('./draft/graphics/compare_images_estimates_over_targets.png', dpi=600)

def compare_models(model_name, image_ids='data/Hyperion/images_visual_check.csv', trait='N', n_outputs=2):
    match trait:
        case 'N':
            band = 0
        case 'LMA':
            band = 1
        case 'Cab':
            band = 2
    if n_outputs==1:
        band=0

    df=pd.read_csv(image_ids)
    ids = df[df['keep']==True]['fnames'].to_list()

    # Create linear regression object
    regr = linear_model.LinearRegression()

    fig, ax = plt.subplots(1, 2, figsize=(FW,0.5*FW))
    for j, id in enumerate(ids):
        result_dir, _,_ = get_path_and_ids(model_name=model_name+id)
        dir_path = [result_dir+'train/', result_dir+'test/']
        for i in range(len(dir_path)):
            targets, preds = get_results(dir_path[i], id, n_outputs, combined=True)
            if 'train' in dir_path[i]:
                cp = cp_train
            else:
                cp = cp_test
            if targets.ndim > 1:
                targets = targets[band]
            if preds.ndim > 1:
                preds = preds[band]
            # Train the model using the training sets
            regr.fit(targets.reshape(-1, 1), preds.reshape(-1, 1))
            x = np.array([np.nanmin(targets), np.nanmax(targets)]).reshape(-1, 1)
            y = regr.predict(x)
            sns.lineplot(ax=ax[i], x=x.squeeze(), y=y.squeeze(), color=cp[-(j+1)])
            if j == (len(ids)-1):
                sns.lineplot(ax=ax[i], x=x.squeeze(), y=x.squeeze(), color='k')
                ax[i].set_xlabel(f'Target {trait.upper()}')
    ax[0].set_title('Training')
    ax[1].set_title('Validation')
        # ax[i].get_legend().remove()
    # fig.suptitle(f'{trait} Estimates over Target Values')
    fig.supylabel(f'Estimated {trait.upper()}')
    fig.tight_layout(pad=2)
    fig.savefig('./draft/graphics/compare_models_estimates_over_targets.png', dpi=600)

def get_path_and_ids(model_name):
    
    result_dir = './data/dbs/'+model_name+'/results/'

    train_ids = []
    for item in os.scandir(result_dir+'train/'):
        if item.is_dir():
            train_ids.append(item.name)
    test_ids = []
    for item in os.scandir(result_dir+'test/'):
        if item.is_dir():
            test_ids.append(item.name)
    
    return result_dir, train_ids, test_ids

def eval_training(dir_path, ids, traits=['cab', 'prot', 'cbc'], model_name='inform_reflectance', n_outputs=2):
    # EO1H0490252009206110K1 EO1H0490252010166110K0
    
    learning_curve('code/uwu_net/saved_models/'+model_name, 
                    title=f'MSE of Synthetic Data for {model_name}', ylim=None, yscale='log', f_path=f'./draft/graphics/learning_curve_{model_name}.svg')
    
    # single image
    for band, trait in enumerate(traits):
        scatter_predictions(dir_path, ids, trait, band, n_outputs, f_path=f'./draft/graphics/scatter_{model_name}_')

def get_species(ids):
    return ['ABAM' if i==1
            else 'ABBA' if i==2
            else 'ABLA' if i==3
            else 'ACMA' if i==4
            else 'ACRU' if i==5
            else 'ACSA' if i==6
            else 'ALIN' if i==7
            else 'ALRU' if i==8
            else 'BEAL' if i==9
            else 'BEPA' if i==10
            else 'CHNO' if i==11
            else 'FRNI' if i==12
            else 'LALA' if i==13
            else 'LAOC' if i==14
            else 'PIAB' if i==15
            else 'PIEN' if i==16
            else 'PIGL' if i==17
            else 'PIMA' if i==18
            else 'PIRU' if i==19
            else 'PISI' if i==20
            else 'PIAL' if i==21
            else 'PIBA' if i==22
            else 'PICO' if i==23
            else 'PIPO' if i==24
            else 'PIRE' if i==25
            else 'PIST' if i==26
            else 'POBA' if i==27
            else 'POGR' if i==28
            else 'POTR' if i==29
            else 'PSME' if i==30
            else 'QURU' if i==31
            else 'THOC' if i==32
            else 'THPL' if i==33
            else 'TSCA' if i==34
            else 'TSHE' if i==35
            else 'TSME' if i==36
            else 'ULAM' if i==37
            else 'none' for i in ids]

def get_units(traits):
    return [     r'$\left[\mu \mathrm{g}.\mathrm{cm}^{-2}\right]$' if t=='cab'
            else r'$\left[\mu \mathrm{g}.\mathrm{cm}^{-2}\right]$' if t=='car'
            else r'$\left[\mu \mathrm{g}.\mathrm{cm}^{-2}\right]$' if t=='ewt'
            else r'$\left[\mu \mathrm{g}.\mathrm{cm}^{-2}\right]$' if t=='prot'
            else r'$\left[\mu \mathrm{g}.\mathrm{cm}^{-2}\right]$' if t=='cbc'
            else r'$\left[\mu \mathrm{g}.\mathrm{cm}^{-2}\right]$' if t=='lma'
            else r'$\left[\mathrm{stem}.\mathrm{ha}^{-1}\right]$' if t=='sd'
            else r'$\left[m\right]' if t=='cd'
            else r'$\left[m\right]' if t=='h'
            else r'$\left[%\right]' if t=='cc'
            else r'$\left[\mathrm{m^2}.\mathrm{m^{-2}}\right]$' if  t=='lai'
            else '' for t in traits]
            
def get_tree_type(s):
    conifers=['ABAM','ABBA','ABLA','CHNO','LALA','LALY','LAOC','PIAB','PIEN','PIGL','PIMA','PIRU','PISI','PIAL','PIBA','PICO','PIPO','PIRE','PIST','PSME','THOC','THPL','TSCA','TSHE','TSME']
    broadleafes=['ACMA','ACRU','ACSA','ALIN','ALRU','BEAL','BEPA','FRNI','POBA','POGR','POTR','QURU','ULAM']
    
    if s in conifers:
        tt = 210
    elif s in broadleafes:
        tt = 220
    else: tt = 0
    return tt

def get_tile_idx(file):
    s = file.rfind('_', )
    e = file.find('.tif', )
    return file[s+1:e]

def eval_spatial(model_name, path_pred=None, path_target='D:/Projects/MA/data/Hyperion/cropped/', tile_id_train=63, tile_id_test=61):
    if path_pred is None:
        path_pred = 'data/Hyperion/'+model_name+'/results_spatial/'
    bands = [f'{d:.0f}'for d in pd.read_csv('data/dbs/bands.csv')['bands']]
    
    # Data Preparation    
    #iterate over all training and validation pairs for target and prediction. Over all individual tiles
    # for each save, Hyperion_id, train/test, tile_id, ssim
    df_ssim = pd.DataFrame()
    for case in ['train', 'test']:
        for path, subdirs, files in os.walk(os.path.join(path_pred, case)):
            for pred_file in files:
                if pred_file.endswith('.tif') and 'prediction' in pred_file:
                    tile_id = get_tile_idx(pred_file)
                    df_ssim = pd.concat([df_ssim, pd.DataFrame({'tile_id':[tile_id], 'case':[case]})], ignore_index=True)
                    target_file = os.path.join(path_target, f'{model_name}_{tile_id}.tif')
                    with rasterio.open(os.path.join(path_pred, case, model_name, pred_file)) as pred_img:
                        pred = pred_img.read()
                    with rasterio.open(target_file) as target_img:
                        target = target_img.read()
                    for band in range(target.shape[0]):
                        ssim_band = ssim(target[band,...].squeeze(), pred[band,...].squeeze())
                        df_ssim.loc[df_ssim['tile_id'] == tile_id, str(band)] = ssim_band

    last_band = int(df_ssim.columns[-1])
    df_describe_train = df_ssim[df_ssim['case']=='train'].describe(percentiles=[.05, .95])
    df_describe_test = df_ssim[df_ssim['case']=='test'].describe(percentiles=[.05, .95])
    median_train = np.median(df_ssim[df_ssim['case']=='train'].loc[:, str(0):str(last_band)], axis=0)
    median_test = np.median(df_ssim[df_ssim['case']=='test'].loc[:, str(0):str(last_band)], axis=0)
    sd_train = np.std(df_ssim[df_ssim['case']=='train'].loc[:, str(0):str(last_band)], axis=0)
    sd_test = np.std(df_ssim[df_ssim['case']=='test'].loc[:, str(0):str(last_band)], axis=0)

    # calc average ssim for individual tiles in training and testing
    # show one individual tile comparison

    # calc image of differences 
    # Tile from Training
    with rasterio.open(os.path.join(path_target, f'{model_name}_{tile_id_train}.tif')) as img:
        target_train = img.read()
    with rasterio.open(os.path.join(path_pred, 'train', model_name, f'prediction_{tile_id_train}.tif')) as img:
        pred_train = img.read()
    mse_band_train = mean_squared_error(target_train[31, ...].squeeze().reshape(1,-1), pred_train[31, ...].squeeze().reshape(1,-1), multioutput='raw_values').reshape(target_train.shape[1:])
    mse_train = mean_squared_error(target_train.reshape(1,1,-1).squeeze(0), pred_train.reshape(1,1,-1).squeeze(0), 
                                    multioutput='raw_values').reshape(target_train.shape)

    # Tile from Validation
    with rasterio.open(os.path.join(path_target, f'{model_name}_{tile_id_test}.tif')) as img:
        target_test = img.read()
    with rasterio.open(os.path.join(path_pred, 'test', model_name, f'prediction_{tile_id_test}.tif')) as img:
        pred_test = img.read()
    mse_band_test = mean_squared_error(target_test[31, ...].squeeze().reshape(1,-1), pred_test[31, ...].squeeze().reshape(1,-1), multioutput='raw_values').reshape(target_test.shape[1:])
    mse_test = mean_squared_error(target_test.reshape(1,1,-1).squeeze(0), pred_test.reshape(1,1,-1).squeeze(0),
                                    multioutput='raw_values').reshape(target_test.shape)

    # Plotting
    learning_curve('code/uwu_net/saved_models/'+model_name, title=f'Learning Curve for Spatial Training on {model_name}', ylim=None, yscale='log', f_path=f'./draft/graphics/learning_curve_{model_name}_spatial.svg')

    # compare mosaics of all tiles and calc ssim over total image
    # fig1 = plt.figure(figsize=(FW,1.22*FW))
    # grid1 = AxesGrid(fig1, 111,
    #             nrows_ncols=(1, 2),
    #             axes_pad=0.2,
    #             share_all=True,
    #             cbar_mode='edge',
    #             cbar_location='right',
    #             cbar_pad=0.1
    #             )
    # with rasterio.open(path_target+'mosaic.tif') as img_t:
    #     i0 = grid1[0].imshow(img_t.read(31), cmap=cmap)
    # with rasterio.open(path_pred+'mosaic.tif') as img_p:
    #     i1 = grid1[1].imshow(img_p.read(31), cmap=cmap)
    
    # # fig1.suptitle(f'Mosaic of {model_name} at {bands[31]} nm')
    # fig1.tight_layout(rect=(0,0,1,0.95))
    # for ax in grid1:
    #     ax.grid(False)
    # grid1.cbar_axes[0].colorbar(i0)
    # grid1.cbar_axes[1].colorbar(i1)

    # grid1[0].set_title('Original')
    # grid1[1].set_title('Reconstruction')
    # fig1.savefig('./draft/graphics/spatial_mosaic.png', dpi=600)
    
    # plot image of differences per pixel
    imgs = [None]*8
    fig2 = plt.figure(figsize=(FW,1.05*FW))
    grid2 = AxesGrid(fig2, 111,
                nrows_ncols=(4, 2),
                axes_pad=0.2,
                share_all=True,
                cbar_mode='edge',
                cbar_location='right',
                cbar_pad=0.1
                )
    min_ref = np.min([target_train[31, ...], target_test[31, ...], pred_train[31, ...], pred_test[31, ...]])
    max_ref = np.max([target_train[31, ...], target_test[31, ...], pred_train[31, ...], pred_test[31, ...]])
    # min_rec = np.min([pred_train[31, ...], pred_test[31, ...]])
    # max_rec = np.max([pred_train[31, ...], pred_test[31, ...]])
    # min_eb = np.quantile([mse_band_train, mse_band_test], 0.01)
    # max_eb = np.quantile([mse_band_train, mse_band_test], 0.99)
    min_e = np.quantile([mse_train, mse_test], 0.01)
    max_e = np.quantile([mse_train, mse_test], 0.99)
    imgs[0] = grid2[0].imshow(target_train[31, ...].squeeze(), vmin=min_ref, vmax=max_ref, cmap=cmap)
    imgs[1] = grid2[2].imshow(pred_train[31, ...].squeeze(), vmin=min_ref, vmax=max_ref, cmap=cmap)
    imgs[2] = grid2[4].imshow(mse_band_train, vmin=min_e, vmax=max_e, cmap=cmap)
    imgs[3] = grid2[6].imshow(np.mean(mse_train, axis=0), vmin=min_e, vmax=max_e, cmap=cmap)
    imgs[4] = grid2[1].imshow(target_test[31, ...].squeeze(), vmin=min_ref, vmax=max_ref, cmap=cmap)
    imgs[5] = grid2[3].imshow(pred_test[31, ...].squeeze(), vmin=min_ref, vmax=max_ref, cmap=cmap)
    imgs[6] = grid2[5].imshow(mse_band_test, vmin=min_e, vmax=max_e, cmap=cmap)
    imgs[7] = grid2[7].imshow(np.mean(mse_test, axis=0), vmin=min_e, vmax=max_e, cmap=cmap)
    labels = ['Reflectance', 'MSE']
    extend = ['neither', 'max']
    for i, axis in enumerate(grid2):
        axis.grid(False)
        if (i+1)%2==0:
            cax = grid2.cbar_axes[i//2].colorbar(imgs[i-1], extend=extend[i//4])
            cax.set_label(labels[i//4])
        else:
            axis.set_xticks(np.arange(target_train.shape[2]+1, step=int(target_train.shape[2]/4)))
            axis.set_yticks(np.arange(target_train.shape[1]+1, step=int(target_train.shape[1]/4)))
    # fig2.suptitle('Comparison of two Tiles')
    fig2.tight_layout(rect=(0,0,1,0.95))
    grid2[0].set_title(f'Training, Tile {tile_id_train}')
    grid2[1].set_title(f'Validation, Tile {tile_id_test}')
    grid2[0].set_ylabel('Original', rotation=0, labelpad=50)
    grid2[2].set_ylabel('Reconstruction', rotation=0, labelpad=50)
    grid2[4].set_ylabel(f'MSE {bands[31]} nm', rotation=0, labelpad=50)
    grid2[6].set_ylabel('Average MSE', rotation=0, labelpad=50)
    fig2.savefig('./draft/graphics/spatial_compare_tiles.png', dpi=600, bbox_inches='tight')

    fig3, ax3 = plt.subplots(figsize=(FW,0.6*FW), sharey=True)
    sns.lineplot(median_train, ax=ax3, color=palette_diverging[0], label=r'$Median_{Train}$')
    sns.lineplot(median_test, ax=ax3, color=palette_diverging[-2], label=r'$Median_{Val}$')
    sns.lineplot(sd_train, ax=ax3, color=palette_diverging[0], linestyle='--', label=r'$\sigma_{Train}$')
    sns.lineplot(sd_test, ax=ax3, color=palette_diverging[-2], linestyle='--', label=r'$\sigma_{Val}$')

    # sns.lineplot(df_describe_test.loc['mean', :str(last_band)]-df_describe_test.loc['std', :str(last_band)], ax=ax3, color=palette_diverging[-2], linestyle='--')
    # sns.lineplot(df_describe_test.loc['mean', :str(last_band)]+df_describe_test.loc['std', :str(last_band)], ax=ax3, color=palette_diverging[-2], linestyle='--', label=r'$\pm 1 \sigma_{Val}$')
    ax3.fill_between(np.arange(last_band+1), 
                    df_describe_train.loc['min', :str(last_band)], df_describe_train.loc['max', :str(last_band)],
                    color=palette_diverging[0], alpha=0.2, label='Range in training data')
    ax3.fill_between(np.arange(last_band+1), 
                    df_describe_test.loc['min', :str(last_band)], df_describe_test.loc['max', :str(last_band)],
                    color=palette_diverging[-2], alpha=0.2, label='Range in Validation data')
    
    # fig3.suptitle(f'Strutural Similarity Index for Hyperion Image {model_name}')
    fig3.tight_layout(rect=(0,0,1,0.95))
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    ax3.set_xlabel('Wavelength in [nm]')
    ax3.set_ylabel('SSIM')
    ax3.set_xlim([0, last_band])
    ax3.set_ylim([0., 1.])
    ax3.set_xticks(np.arange(last_band+1)[::4])
    ax3.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax3.set_xticklabels(bands[::4], rotation=90)
    fig3.savefig('./draft/graphics/spatial_ssim.svg', bbox_inches='tight')

def load_secondary_maps(model_name, theta_cc=None, theta_conf=None, theta_lc=None, lct_sp=None, path_pred=None):
    '''Load estimates and call function to filter estimates.'''
    if path_pred is None:
        path_pred = 'data/Hyperion/'+model_name+'/results/'
    tiles=[]
    for path, subdirs, files in os.walk(path_pred):
        for pred_file in files:
            if pred_file.endswith('.tif') and 'prediction' in pred_file:
                tiles.append(os.path.join(path, pred_file))

    with rasterio.open(path_pred+'mosaic.tif') as img:
        estimates = img.read()

    estimates, species = filter_array(estimates, model_name=model_name, theta_cc=theta_cc, theta_conf=theta_conf, theta_lc=theta_lc, lct_sp=lct_sp)
    
    return estimates, species

def ssim_matrix(estimates, traits):
    '''
    Calculate ssim between very possible combination of trait pairings, cumulative for all Tiles
    Return ssim_matrix as commulative 
    '''
    grid = np.meshgrid(np.arange(len(traits)), np.arange(len(traits)))
    ssim_arr = np.empty(shape=(len(traits), len(traits)))
    for pair in tqdm(zip(grid[0].reshape(1,-1).squeeze(), grid[1].reshape(1,-1).squeeze())):
        ssim_arr[pair[0], pair[1]] = ssim(estimates[pair[0],:].squeeze(), estimates[pair[1],:].squeeze())
    return ssim_arr

def correlation_matrix(estimates, traits, title='', 
                    subtitles=['Spectral-Spatial Training', 'Only Spectral Training']):
                    # 'SSIM Between Traits for Spatial-Spectral and only Spectral Training'
    mat=[]
    
    fig = plt.figure(figsize=(FW,0.5*FW))
    grid = AxesGrid(fig, 111,
                nrows_ncols=(1, len(estimates)),
                axes_pad=0.2,
                share_all=True,
                cbar_pad=0.2
                )
    for e, est in enumerate(estimates):
        mat.append(ssim_matrix(est, traits))
        heatmap = sns.heatmap(mat[e], ax=grid[e], vmin=0, vmax=1, 
                    xticklabels=traits, yticklabels=[t.upper() for t in traits], 
                    annot=True, cmap=cmap,
                    cbar=False)
        grid[e].set_title(subtitles[e])
        for label in grid[e].get_yticklabels():
            label.set_rotation(0)
    # grid.cbar_axes[-1].colorbar(norm=None, cmap=cmap)
    # fig.suptitle(title)
    fig.tight_layout(rect=(0,0,1,0.95))
    fig.savefig('./draft/graphics/correlation_matrices.svg')

def species_analysis(ests, traits, species):
    '''
    calculate the estimates per species as distribution
    '''
    isna = lambda estimate: estimate[~np.isnan(estimate)]
    df_e = pd.DataFrame()
    traintype = ['Spectral-Spatial', 'Only Spectral']
    for e, estimates in enumerate(ests):
        estimates[estimates==0.] = np.nan
        d_traits = {}
        for sp in np.unique(species):
            d_traits[sp] = {trait: isna(estimates[t, species==sp].squeeze()) for t, trait in enumerate(traits)}
        df = pd.DataFrame(d_traits).transpose().drop(0.0)
        df['species'] = df.index
        df = df.set_index(['species']).apply(pd.Series.explode).reset_index()
        df.dropna(inplace=True)
        df = df.apply(pd.to_numeric)
        counts = df['species'].value_counts()
        df = df[~df['species'].isin(counts[counts < 20].index)]
        df['traintype'] = traintype[e]
        df_e = pd.concat([df_e, df], ignore_index=True)
    df_species = df_e.groupby('species')
    df_e[df_e['traintype']=='Spectral-Spatial'].groupby('species').count()
    summary = df_species.describe(percentiles=[0.025, 0.975])

    units = get_units(traits)
    tt = [get_tree_type(s) for s in get_species(df_e['species'].unique())]
    colors = [palette[5], palette[0],palette[-1], palette[-6]]
    pal = []
    for lct in tt:
        if lct==210:
            pal.extend(colors[0:2])
        else:
            pal.extend(colors[2:])

    # Calculate T-test H_0: estimated trait is the same for broadleaf (29) and coniferous
    for ty in traintype:
        print(f'T-Test: H_0 is mu_29 == mu_not29, {ty}')
        for t in traits:
            print(t, ttest_ind(df_e.loc[(df_e['traintype']==ty) & (df_e['species']==29)][t].sample(n=1000), df_e.loc[(df_e['traintype']==ty) & (df_e['species']!=29)][t].sample(n=1000), equal_var=False))
    print('Z-Test: H_0 is mu_29 == mu_not29')
    for t in traits:
        compare_mean = ws.CompareMeans(ws.DescrStatsW(df_e[df_e['species']==29][t].sample(n=10000).values), ws.DescrStatsW(df_e[df_e['species']!=29][t].sample(n=10000).values))
        print(t, compare_mean.ztest_ind(usevar='unequal'))

    # Plotting
    fig, axes = plt.subplots(int(len(traits)/2), 2, figsize=(FW, 0.6*FW), sharex=True)
    for t, ax in enumerate(axes.flatten()):
        bp = sns.boxplot(df_e, x='species', y=traits[t], hue=df_e['traintype'],
            showfliers = False, flierprops={"marker": "x"},
            whis=[5, 95], showcaps=False,
            linewidth=0.5,
            medianprops={"color": "coral"},
            orient='v', ax=ax
        )
        bp.set_xlabel('')
        bp.set_ylabel(f'{traits[t].upper()} {units[t]}', loc='center', rotation=90, labelpad=10)
        ax.get_legend().remove()
        boxes = ax.findobj(mpl.patches.PathPatch)
        for color, box in zip(pal, boxes):
            box.set_facecolor(color)
    axes[1,0].set_xticklabels(get_species(df_e['species'].unique()), rotation=90)
    axes[1,1].set_xticklabels(get_species(df_e['species'].unique()), rotation=90)
    axes[1,0].set_xlabel('Species')
    axes[1,1].set_xlabel('Species')
    fig.tight_layout(rect=(0,0,1,0.95))
    # fig.suptitle('Estimated Trait per Species')
    legend_elements = [Patch(facecolor=colors[0], label=f'{traintype[0]}, coniferous'),
                        Patch(facecolor=colors[1], label=f'{traintype[1]}, coniferous'),
                        Patch(facecolor=colors[2], label=f'{traintype[0]}, broadleafed'),
                        Patch(facecolor=colors[3], label=f'{traintype[1]}, broadleafed')]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.savefig('./draft/graphics/species_analysis.svg')

def eval_real_prediction(model_name, traits, tile_id=61, path_spectral=None):
    path_pred = 'data/Hyperion/'+model_name+'/results/'
    make_mosaic(path_pred, 'prediction')
    make_mosaic(path_spectral, 'prediction')
    
    with rasterio.open(path_pred+'mosaic.tif') as img:
        arr = img.read()
    with rasterio.open(path_spectral+'mosaic.tif') as img_spec:
        arr_spec = img_spec.read()
    with rasterio.open('./data/Hyperion/'+model_name+'_Reflectance_topo_v2_Reflectance.img') as img:
        reflectance = img.read(14).squeeze()
    offset = [int((reflectance.shape[i]-arr.shape[i+1])/2) for i in [0,1]]
    reflectance = reflectance[offset[0]:-offset[0]-1, offset[1]:-offset[1]-1]
    arr[:, reflectance==0.] = 0.
    arr_spec[:, reflectance==0.] = 0.

    # Plot moasic image of first trait
    fig1 = plt.figure(figsize=(FW,1.5*FW))
    grid1 = ImageGrid(fig1, 111,
                nrows_ncols=(1, 2),
                axes_pad=0.5,
                # share_all=True,
                cbar_mode='each',
                cbar_location='right',
                cbar_pad=0.1
                )

    img = grid1[0].imshow(arr[0,...].squeeze(), cmap=cmap)
    img_spec = grid1[1].imshow(arr_spec[0,...].squeeze(), cmap=cmap)
    # fig1.suptitle(f'Mosaic of Estimated {traits[0]}')
    fig1.tight_layout(rect=(0,0,1,0.95))
    for ax in grid1:
        ax.grid(False)
    
    grid1.cbar_axes[0].colorbar(img, label=traits[0])
    # grid1.cbar_axes[0].axis[grid1.cbar_axes[0].orientation].label.set_text(traits[0])
    grid1.cbar_axes[1].colorbar(img_spec)
    # grid1.cbar_axes[1].axis[grid1.cbar_axes[1].orientation].label.set_text(traits[0])
    grid1[0].set_title('Spectral-Spatial Training')
    grid1[1].set_title('Only Spectral Training')
    fig1.savefig(f'./draft/graphics/mosaic_estimates_{traits[0]}.png', dpi=600)

    # Compare with no_spatial model
    with rasterio.open(os.path.join(path_pred, 'test', model_name, f'prediction_{tile_id}.tif')) as img:
        arr = img.read()
    with rasterio.open(os.path.join(path_spectral, 'test', model_name, f'prediction_{tile_id}.tif')) as img_spec:
        arr_spec = img_spec.read()
    with rasterio.open(f'./data/Hyperion/cropped/{model_name}_{tile_id}.tif') as img:
        reflectance = img.read(14).squeeze()
    arr[:, reflectance==0.] = 0
    arr_spec[:, reflectance==0.] = 0

    diff = arr - arr_spec

    units = get_units(traits)

    imgs = [None]*len(traits)*2
    fig2 = plt.figure(figsize=(4/3*FW,1.05*4/3*FW))
    grid2 = ImageGrid(fig2, 111,
                nrows_ncols=(len(traits), 2),
                axes_pad=0.1,
                share_all=True,
                cbar_mode='edge',
                cbar_location='right',
                cbar_pad=0.2
                )
    for t in range(len(traits)):
        vmin = np.min([arr[t,...], arr_spec[t,...]])
        vmax = np.max([arr[t,...], arr_spec[t,...]])
        imgs[t] = grid2[2*t].imshow(arr[t, ...].squeeze(), vmin=vmin, vmax=vmax, cmap=cmap_traits)
        imgs[t+len(traits)] = grid2[2*t+1].imshow(arr_spec[t, ...].squeeze(), vmin=vmin, vmax=vmax, cmap=cmap_traits)

    for i, axis in enumerate(grid2):
        axis.grid(False)
        # axis.set_title(f'{traits[i//2].upper()} {units[i//2]}', pad=10)
        if (i+1)%2==0:
            sm = mpl.cm.ScalarMappable(
                norm = mpl.colors.Normalize(np.min([arr[i//2,...], arr_spec[i//2,...]]), np.max([arr[i//2,...], arr_spec[i//2,...]])), 
                cmap = cmap_traits)
            cax = grid2.cbar_axes[i//2].colorbar(sm)
            cax.set_label(f'{traits[i//2].upper()} {units[i//2]}', rotation=90, labelpad=10, verticalalignment='bottom')
        else:
            axis.set_xticks([]) # np.arange(arr.shape[2]+1, step=int(arr.shape[2]/4)))
            axis.set_yticks([]) # np.arange(arr.shape[1]+1, step=int(arr.shape[1]/4)))
            # grid2[i].set_ylabel(f'{traits[i//2]}\n{units[i//2]}', rotation=0, labelpad=20)
    fig2.tight_layout(rect=(0,0,1,0.95))
    # fig2.suptitle(f'Comparison of Estimated Traits on Tile {tile_id}')
    grid2[0].set_title('Spectral-Spatial')
    grid2[1].set_title('Only Spectral')
    fig2.savefig('./draft/graphics/compare_trait_tile.png', dpi=600, bbox_inches='tight')

    imgs = [None]*len(traits)
    fig3 = plt.figure(figsize=(2/3*FW,2.1*2/3*FW))
    grid3 = ImageGrid(fig3, 111,
                nrows_ncols=(len(traits), 1),
                axes_pad=0.1,
                share_all=True,
                cbar_mode='each',
                cbar_location='right',
                cbar_pad=0.2
                )

    for i, axis in enumerate(grid3):
        vmin = np.quantile(diff[i, ...], 0.01)
        vmax = np.quantile(diff[i, ...], 0.99)
        diff[i, diff[i, ...]<vmin] = vmin
        diff[i, diff[i, ...]>vmax] = vmax
        half_range = np.max([np.abs(vmin), np.abs(vmax)])
        imgs[i] = grid3[i].imshow(diff[i, ...].squeeze(), vmin=-half_range, vmax=half_range, cmap=cmap_diverging)
        axis.grid(False)
        cax = grid3.cbar_axes[i].colorbar(imgs[i], extend='both')
        cax.set_ticks([-half_range, 0.0, half_range])
        labels = [f'-{half_range:.2e}', '0.00', f'{half_range:.2e}'] if half_range<0.1 else [f'-{half_range:.2f}', '0.00', f'{half_range:.2f}']
        cax.set_ticklabels(labels)
        cax.set_label(f'{traits[i].upper()} {units[i]}', rotation=90, labelpad=10)
        # axis.set_title(f'{traits[i].upper()} {units[i]}', pad=10)
        axis.set_xticks([]) # np.arange(diff.shape[2]+1, step=int(diff.shape[2]/4)))
        axis.set_yticks([]) # np.arange(diff.shape[1]+1, step=int(diff.shape[1]/4)))
        # grid3[i].set_ylabel(f'{traits[i]}\n{units[i]}', rotation=0, labelpad=20)
    fig3.tight_layout(rect=(0,0,1,0.95))
    grid3[0].set_title('Difference')
    fig3.savefig('./draft/graphics/trait_differences.png', dpi=600)

def trait_compare(model_name, traits, tile_id=38):
    path_pred = 'data/Hyperion/'+model_name+'/results/'
    with rasterio.open(os.path.join(path_pred, 'train', model_name, f'prediction_{tile_id}.tif')) as img:
        arr = img.read()
    with rasterio.open(f'./data/Hyperion/cropped/{model_name}_{tile_id}.tif') as img:
        r = img.read(16)
        g = img.read(5)
        b = img.read(1)
        rgb = np.dstack((r,g,b))
    arr[:, g==0.] = 0

    scaler = MinMaxScaler()
    rgb = scaler.fit_transform(rgb.reshape(-1, 3)).reshape(rgb.shape)
    units = get_units(traits)

    fig1, ax1 = plt.subplots(figsize=(0.6*FW, 0.3*FW))
    ax1.imshow(rgb)
    ax1.set_xticks(np.arange(rgb.shape[1]+1, step=int(rgb.shape[1]/4)))
    ax1.set_yticks(np.arange(rgb.shape[0]+1, step=int(rgb.shape[0]/4)))
    ax1.grid(False)
    fig1.savefig(f'./draft/graphics/tile_{tile_id}.png', dpi=600)

    imgs = [None]*len(traits)
    fig2 = plt.figure(figsize=(FW,0.6*FW))
    grid2 = ImageGrid(fig2, 111,
                nrows_ncols=(int(len(traits)/2), 2),
                axes_pad=(0.7, 0.3),
                share_all=True,
                cbar_mode='each',
                cbar_location='right',
                cbar_pad=0.1
                )
    for t in range(len(traits)):
        imgs[t] = grid2[t].imshow(arr[t, ...].squeeze(), cmap=cmap_traits)
        grid2[t].grid(False)
        cax =grid2.cbar_axes[t].colorbar(imgs[t])
        cax.set_label(f'{units[t]}', rotation=90)

        grid2[t].set_xticks(np.arange(arr.shape[2]+1, step=int(arr.shape[2]/4)))
        grid2[t].set_yticks(np.arange(arr.shape[1]+1, step=int(arr.shape[1]/4)))
        grid2[t].set_title(f'{traits[t].upper()}')
    fig2.tight_layout(rect=(0,0,1,0.95))
    # fig2.suptitle(f'Comparison of Estimated Traits on Tile {tile_id}')
    fig2.savefig('./draft/graphics/traits_tile.png', dpi=600)

def f_float(x):
    return '%.2f' % x

def f_sci(x):
    return '%.2e' % x

# %% --------------------------------------------------------------------------------------
# settings
model_name = 'EO1H0450222004225110PZ'

result_dir, train_ids, test_ids = get_path_and_ids(model_name='inform_'+model_name)
dir_path = [result_dir+'train/', result_dir+'test/']
ids = [train_ids, test_ids]
# print(ids)

# set all colors
cmap = sns.color_palette("mako", as_cmap=True)
cmap_traits = mpl.colors.LinearSegmentedColormap.from_list('cm_traits',
    plt.get_cmap('terrain_r')(np.linspace(0.25, 0.75, 256)))
palette = sns.color_palette("crest_r", 20)
palette_diverging = sns.color_palette("PuOr_r", 11)
cmap_diverging = sns.color_palette("icefire", 11, as_cmap=True)
cp_train = sns.color_palette('Blues', 2)# len(ids[0]))
cp_test = sns.color_palette('YlOrBr', 2)# len(ids[1]))
c_train = cp_train[int(len(cp_train)/2)]
c_test = cp_test[int(len(cp_test)/2)] 
sns.set()
sns.set_style("whitegrid")
sns.despine()
sns.set_context("paper", rc={"font.size":11,"axes.titlesize":11,"axes.labelsize":11})#, "text.usetex":True, 'text.latex.preamble':r'\usepackage{amsmath}'})

traits_all = ['cab','car','ewt','prot','cbc','lma','lai', 'cc']
traits = ['cab','prot','cbc','lai']
n_outputs=len(traits)

# %%
# eval training

# # Combined model
# # learning curve and scatter plots
# eval_training(dir_path, ids, traits=traits, model_name='inform_reflectance', n_outputs=n_outputs)

# # compare model across images
# compare_images(dir_path, ids, trait=traits[0], n_outputs=n_outputs)
# compare_images(dir_path, ids, trait=traits[1], n_outputs=n_outputs)

# # compare individual models
# compare_models('inform_', trait=traits[0], n_outputs=n_outputs)
# compare_models('inform_', trait=traits[1], n_outputs=n_outputs)

mns =  [model_name]
for mn in mns:
    result_dir, train_ids, test_ids = get_path_and_ids(model_name='inform_'+mn)

    result_dir = 'data/dbs/inform_'+model_name+'/results/'

    dir_path = [result_dir+'train/', result_dir+'test/']
    ids = [train_ids, test_ids]
    eval_training(dir_path, ids, traits=traits, model_name='inform_'+mn, n_outputs=n_outputs)

    results = eval('inform_'+mn, traits, case='train', dir_path=result_dir)
    results = pd.concat([results, eval('inform_'+mn, traits, case='test', dir_path=result_dir)], ignore_index=True)
    results['Trait'] = results['Trait'].str.upper()
    results['Case'] = results['Case'].str.capitalize()
    results.set_index(pd.MultiIndex.from_frame(results.loc[:,['Case','Trait']]), inplace=True)
display(results)
# results.drop(columns=['Trait', 'Case']).iloc[:, 2:].to_latex('draft/tables/results_spec_spat.tex',  index=True,
#             formatters=[f_float, f_sci, f_float], sparsify=True, column_format='@{}ccrrr@{}', multirow=True,
#             header=['$R^2$', 'MAE', 'NRMSE'], encoding='utf-8', escape=False)

for mn in mns:
    result_dir = 'data/dbs/inform_'+model_name+'/results_8_traits/'

    dir_path = [result_dir+'train/', result_dir+'test/']
    ids = [train_ids, test_ids]
    eval_training(dir_path, ids, traits=traits_all, model_name='inform_'+mn+'_8_traits', n_outputs=len(traits_all))

    results = eval('inform_'+mn+'_8_traits', traits_all, case='train', dir_path=result_dir)
    results = pd.concat([results, eval('inform_'+mn+'_8_traits', traits_all, case='test', dir_path=result_dir)], ignore_index=True)
    results['Trait'] = results['Trait'].str.upper()
    results['Case'] = results['Case'].str.capitalize()
    results.set_index(pd.MultiIndex.from_frame(results.loc[:,['Case','Trait']]), inplace=True)
display(results)
# results.drop(columns=['Trait', 'Case']).iloc[:, 2:].to_latex('draft/tables/results_8_traits.tex',  index=True,
#             formatters=[f_float, f_sci, f_float], sparsify=True, column_format='@{}ccrrr@{}', multirow=True,
#             header=['$R^2$', 'MAE', 'NRMSE'], encoding='utf-8', escape=False)
# %%
# only spectral training
for mn in mns:
    result_dir, train_ids, test_ids = get_path_and_ids(model_name='inform_'+mn)

    result_dir = 'data/dbs/inform_'+model_name+'/results_spectral/'

    dir_path = [result_dir+'train/', result_dir+'test/']
    ids = [train_ids, test_ids]
    eval_training(dir_path, ids, traits=traits, model_name='inform_'+mn+'_spectral', n_outputs=n_outputs)

    results = eval('inform_'+mn+'_spectral', traits, case='train', dir_path=result_dir)
    results = pd.concat([results, eval('inform_'+mn+'_spectral', traits, case='test', dir_path=result_dir)], ignore_index=True)
    results['Trait'] = results['Trait'].str.upper()
    results['Case'] = results['Case'].str.capitalize()
    results.set_index(pd.MultiIndex.from_frame(results.loc[:,['Case','Trait']]), inplace=True)
display(results)
# results.drop(columns=['Trait', 'Case']).iloc[:, 2:].to_latex('draft/tables/results_spec.tex',  index=True,
#             formatters=[f_float, f_sci, f_float], sparsify=True, column_format='@{}ccrrr@{}', multirow=True,
#             header=['$R^2$', 'MAE', 'NRMSE'], encoding='utf-8', escape=False)
# %% 
# # eval optuna
#path_optuna_study = 'code/uwu_net/saved_models/optuna/study.pkl'
#optuna_analyse_p(path_optuna_study)

# %%
# eval spatial
# make mosaic of train and val combined for predictions
path_target='D:/Projects/MA/data/Hyperion/cropped/'
path_pred = 'data/Hyperion/'+model_name+'/results_spatial/'

make_mosaic(path_target, model_name, dst_path=path_target+'mosaic.tif')
make_mosaic(path_pred, 'prediction', dst_path=path_pred+'mosaic.tif')
eval_spatial(model_name=model_name, path_target=path_target, tile_id_train=11, tile_id_test=14)

# %%
# comparison between spatial and only spectral training
sp_class={'conifers':[1,2,3,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,30,32,33,34,35,36],
        'broadleafes':[4,5,6,7,8,9,10,12,27,28,29,31,37]}

conifers=['ABAM','ABBA','ABLA','CHNO','LALA','LALY','LAOC','PIAB','PIEN','PIGL','PIMA','PIRU','PISI','PIAL','PIBA','PICO','PIPO','PIRE','PIST','PSME','THOC','THPL','TSCA','TSHE','TSME']
broadleafes=['ACMA','ACRU','ACSA','ALIN','ALRU','BEAL','BEPA','FRNI','POBA','POGR','POTR','QURU','ULAM']
missing=['LALY']

path_spectral = 'data/Hyperion/'+model_name+'/results_spectral/'

eval_real_prediction(model_name, traits, tile_id=14, path_spectral=path_spectral)
trait_compare(model_name, traits, tile_id=38)

estimates_spec_spat, species = load_secondary_maps(model_name, theta_cc=0.6, theta_conf=0.5, theta_lc=[210,220], lct_sp=sp_class)
estimates_spec, _ = load_secondary_maps(model_name, theta_cc=0.6, theta_conf=0.5, theta_lc=[210,220], lct_sp=sp_class, path_pred=path_spectral)
correlation_matrix([estimates_spec_spat, estimates_spec], traits)

species_analysis([estimates_spec_spat, estimates_spec], traits, species)

# %%
# calculated clipped mosaic
with rasterio.open('D:/Projects/MA/data/Hyperion/cropped/mosaic.tif') as src:
    profile = src.profile
    profile.update(
        driver='GTiff',
        width=estimates_spec_spat.shape[2],
        height=estimates_spec_spat.shape[1],
        count=estimates_spec_spat.shape[0],
        dtype=estimates_spec_spat.dtype
    )
    ref = src.read(20)

with rasterio.open('D:/Projects/MA/data/Hyperion/EO1H0450222004225110PZ/results/estimates_clipped.tif', 'w', **profile) as dst:
    for l in range(estimates_spec_spat.shape[0]):
        dst.write(estimates_spec_spat[l, ...], l+1) 

with rasterio.open('D:/Projects/MA/data/Hyperion/EO1H0450222004225110PZ/results_spatial/mosaic.tif') as im_ref:
    recon = im_ref.read()

recon[:, ref == 0] = 0
profile.update(count=recon.shape[0])
with rasterio.open('D:/Projects/MA/data/Hyperion/cropped/reconstruction_clipped.tif', 'w', **profile) as im_rec:
    for l in range(recon.shape[0]):
        im_rec.write(recon[l, ...], l+1)
        


# %% 
# get bounding box polygon from raster1578590
#path = os.path.join('data/Hyperion/'+model_name+'/results/', 'train', model_name, f'prediction_{38}')
# ra = rasterio.open(path+'.tif')
# bounds  = ra.bounds
# schema = {'geometry': 'Polygon', 'properties': {}}
# df = gpd.GeoDataFrame({"id":1,"geometry":[box(*bounds)]})
# df.to_file(path+'.shp')
# with fiona.open(path+'.shp', 'w', driver='ESRI Shapefile',
#                 crs=ra.crs.to_dict(), schema=schema) as c:
#     c.write({'geometry': mapping(box(*bounds)), 'properties': {}})