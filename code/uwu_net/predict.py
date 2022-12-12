import argparse
from re import S
import fnet.data
import importlib
import joblib
import json
import numpy as np
import os
import pandas as pd
import pdb
import rasterio
import sys
import tifffile
import time
import torch
import warnings

sys.path.append('../../code/')
from utils import id_from_path, undo_scaling, make_mosaic

def set_warnings():
    warnings.filterwarnings('ignore', message='.*zoom().*')
    warnings.filterwarnings('ignore', message='.*end of stream*')
    warnings.filterwarnings('ignore', message='.*multiple of element size.*')

def get_dataset(opts, propper, validation=False):
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]
    transform_signal.append(propper)
    transform_target.append(propper)
    if 'test' in opts.path_dataset_csv:
        validation = True
    ds = getattr(fnet.data, opts.class_dataset)(
        path_csv = opts.path_dataset_csv,
        transform_source = transform_signal,
        transform_target = transform_target,
        path_model = opts.path_model_dir[0],
        scale_signal = opts.scale_signal,
        scale_target = opts.scale_target if not opts.no_target else None,
        validation = validation,
        start = opts.read_channel_from
    )
    print(ds)
    return ds

def save_tiff_and_log(tag, ar, path_tiff_dir, entry, path_log_dir, profile):
    if not os.path.exists(path_tiff_dir):
        os.makedirs(path_tiff_dir)
    path_tiff = os.path.join(path_tiff_dir, '{:s}.tif'.format(tag))
    profile.update(count = ar.shape[0])
    with rasterio.open(path_tiff, 'w', **profile) as tif:
        for l in range(ar.shape[0]):
            tif.write(ar[l, ...], l+1) 
    # tifffile.imwrite(path_tiff, ar)
    print('saved:', path_tiff)
    entry['path_' + tag] = os.path.relpath(path_tiff, path_log_dir)

def get_prediction_entry(dataset, index):
    info = dataset.get_information(index)
    # In the case where 'path_signal', 'path_target' keys exist in dataset information,
    # replace with 'path_signal_dataset', 'path_target_dataset' to avoid confusion with
    # predict.py's 'path_signal' and 'path_target'.
    if isinstance(info, dict):
        if 'path_signal' in info:
            info['path_signal_dataset'] = info.pop('path_signal')
        if 'path_target' in info:
            info['path_target_dataset'] = info.pop('path_target')
        return info
    if isinstance(info, str):
        return {'information': info}
    raise AttributeError

def get_id(entry):
    start = entry['path_signal_dataset'].find('EO1')
    return entry['path_signal_dataset'][start:start+22]

def get_tile_idx(entry):
    s = entry['path_signal_dataset'].rfind('_', )
    e = entry['path_signal_dataset'].find('.tif', )
    return entry['path_signal_dataset'][s:e]
    
def main():
    # set_warnings()
    factor_yx = 1.0  # 0.108 um/px -> 0.29 um/px changed to 1.0 12_03_2019
    default_resizer_str = 'fnet.transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_dataset', default='CziDataset', help='Dataset class')
    parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
    parser.add_argument('--module_fnet_model', default='fnet_model', help='module with fnet_model')
    parser.add_argument('--n_images', type=int, default=16, help='max number of images to test')
    parser.add_argument('--no_prediction', action='store_true', help='set to not save prediction image')
    parser.add_argument('--no_prediction_unpropped', action='store_true', help='set to not save unpropped prediction image')
    parser.add_argument('--no_signal', action='store_true', help='set to not save signal image')
    parser.add_argument('--no_target', action='store_true', help='set to not save target image')
    parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing Dataset')
    parser.add_argument('--path_model_dir', nargs='+', default=[None], help='path to model directory')
    parser.add_argument('--path_save_dir', help='path to output directory')
    parser.add_argument('--propper_kwargs', type=json.loads, default={}, help='path to output directory')
    parser.add_argument('--transform_signal', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset signal')
    parser.add_argument('--transform_target', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset target'),
    parser.add_argument('--scale_signal', default=None, help='scikit-leran scaler for signal'),
    parser.add_argument('--scale_target', default=None, help='scikit-leran scaler for target'),
    parser.add_argument('--read_channel_from', type=int, default=15, help='start reading signal from channel'),
    parser.add_argument('--no_unscale', action='store_true', help='set to save scaled values'),
    parser.add_argument('--final_chan', type=int, default=1, help='number of target parameters'),
    parser.add_argument('--no_idx', action='store_true', help='set to prevent idx in path for tiff'),
    parser.add_argument('--merge_output', action='store_true', help='set to merge output slices to single file'),
    opts = parser.parse_args()

    if os.path.exists(opts.path_save_dir):
        print('Output path already exists.')
        # return #prevent overwriting of previous results
    if opts.class_dataset == 'TiffDataset':
        if opts.propper_kwargs.get('action') == '-':
            opts.propper_kwargs['n_max_pixels'] = 60000000
    propper = fnet.transforms.Propper(**opts.propper_kwargs)
    print(propper)
    model = None
    dataset = get_dataset(opts, propper)
    entries = []
    indices = range(len(dataset)) if opts.n_images < 0 else range(min(opts.n_images, len(dataset)))
    for idx in indices:
        entry = get_prediction_entry(dataset, idx)
        profile = dataset.get_profile(idx)
        #print('**********************TESTING PRINT******************************')
        #print((dataset[0])[0].size())#
        #print((dataset[0])[1].size())#
        data = [torch.unsqueeze(d, 0) for d in dataset[idx]]  # make batch of size 1
        signal = data[0]
        target = data[1] if (len(data) > 1) else None
        path_tiff_dir = os.path.join(opts.path_save_dir, get_id(entry), '')
        if not opts.no_idx:
            idx_tag = get_tile_idx(entry)
        else:
            idx_tag = ''
        for path_model_dir in opts.path_model_dir:
            if (path_model_dir is not None) and (model is None or len(opts.path_model_dir) > 1):
                model = fnet.load_model(path_model_dir, opts.gpu_ids, module=opts.module_fnet_model)
                # model.net.final_chan = opts.final_chan
                print(model)
                name_model = os.path.basename(path_model_dir)
            prediction = model.predict(signal) if model is not None else None
            
            # Unscale predicitons
            if not opts.no_unscale and (opts.scale_signal is not None or opts.scale_target is not None):
                print('undoing scaling')
                id = id_from_path(entry['path_signal_dataset'])
                if 'test' in opts.path_dataset_csv:
                    ds_type = 'train' #'test'
                else:
                    ds_type = 'train'
                if opts.scale_signal is not None:
                    path_signal_scaler = '/'.join([opts.path_model_dir[0],'scaler',ds_type,id+'_'+'signal'+'_scaler.pkl'])
                    if os.path.exists(path_signal_scaler) and opts.scale_signal is not None:
                        signal_scaler = joblib.load(path_signal_scaler)
                        signal = undo_scaling(signal_scaler, signal.numpy()[0, ])
                if opts.scale_target is not None:
                    path_target_scaler = '/'.join([opts.path_model_dir[0],'scaler',ds_type,id+'_'+'target'+'_scaler.pkl'])
                    if os.path.exists(path_target_scaler) and opts.scale_target is not None:
                        target_scaler = joblib.load(path_target_scaler)
                        if not opts.no_target:
                            target = undo_scaling(target_scaler, target.numpy()[0, ])
                        if model.net.final_chan == 1:
                            prediction = undo_scaling(target_scaler, prediction.numpy())
                        else:
                            prediction = undo_scaling(target_scaler, prediction.numpy()[0, ])
            else:
                signal =signal.numpy()[0, ]
                target = target.numpy()[0, ]
                if model.net.final_chan == 1:
                    prediction = prediction.numpy()
                else:
                    prediction = prediction.numpy()[0, ]
            if not opts.no_prediction and prediction is not None:
                save_tiff_and_log('prediction'+idx_tag, prediction, path_tiff_dir, entry, opts.path_save_dir, profile) #_{:s}'.format(name_model)
            if not opts.no_prediction_unpropped:
                ar_pred_unpropped = propper.undo_last(prediction[0, ])
                save_tiff_and_log('prediction_{:s}_unpropped'.format(name_model)+idx_tag, ar_pred_unpropped, path_tiff_dir, entry, opts.path_save_dir, profile)
        entries.append(entry)
        if not opts.no_signal:
            save_tiff_and_log('signal'+idx_tag, signal, path_tiff_dir, entry, opts.path_save_dir, profile)
        if not opts.no_target and target is not None:
            save_tiff_and_log('target'+idx_tag, target, path_tiff_dir, entry, opts.path_save_dir, profile)
   
    with open(os.path.join(opts.path_save_dir, 'predict_options.json'), 'w') as fo:
        json.dump(vars(opts), fo, indent=4, sort_keys=True)
    pd.DataFrame(entries).to_csv(os.path.join(opts.path_save_dir, 'predictions.csv'), index=False)

    if opts.merge_output: 
        make_mosaic(opts.path_save_dir+f'/{get_id(entry)}/', 'prediction')

if __name__ == '__main__':
    main()
