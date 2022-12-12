import argparse
import fnet.data
import fnet.fnet_model
import json
import logging
import numpy as np
import os
import pdb
import sys
import time
import torch
import warnings

sys.path.append('../../code/')
from utils import make_scaler

def get_dataloader(remaining_iterations, opts, validation=False):
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]
    ds = getattr(fnet.data, opts.class_dataset)(
        path_csv = opts.path_dataset_csv if not validation else opts.path_dataset_val_csv,
        transform_source = transform_signal,
        transform_target = transform_target,
        path_model = opts.path_run_dir,
        scale_signal = opts.scale_signal,
        scale_target = opts.scale_target,
        validation = validation,
        start = opts.read_channel_from
    )
    print(ds)
    ds_patch = fnet.data.BufferedPatchDataset(
        dataset = ds,
        patch_size = opts.patch_size,
        final_chan=opts.final_chan,
        buffer_size = opts.buffer_size if not validation else len(ds),
        buffer_switch_frequency = opts.buffer_switch_frequency if not validation else -1,
        npatches = remaining_iterations*opts.batch_size if not validation else 4*opts.batch_size,
        verbose = True,
        shuffle_images = opts.shuffle_images,
        **opts.bpds_kwargs,
    )
    dataloader = torch.utils.data.DataLoader(
        ds_patch,
        batch_size = opts.batch_size,
        drop_last = True, #####################################
    )
    return dataloader

def main():
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser()
    factor_yx = 1.0  # 0.108 um/px -> 0.29 um/px changed to 1.0 12_03_2019
    default_resizer_str = 'fnet.transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch')
    parser.add_argument('--bpds_kwargs', type=json.loads, default={}, help='kwargs to be passed to BufferedPatchDataset')
    parser.add_argument('--buffer_size', type=int, default=5, help='number of images to cache in memory')
    parser.add_argument('--buffer_switch_frequency', type=int, default=720, help='BufferedPatchDataset buffer switch frequency')
    parser.add_argument('--class_dataset', default='CziDataset', help='Dataset class')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
    parser.add_argument('--interval_save', type=int, default=50, help='iterations between saving log/model')
    parser.add_argument('--iter_checkpoint', nargs='+', type=int, default=[], help='iterations at which to save checkpoints of model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_iter', type=int, default=50, help='number of training iterations')
    parser.add_argument('--nn_kwargs', type=json.loads, default={}, help='kwargs to be passed to nn ctor')
    parser.add_argument('--nn_module', default='fnet_nn_UwU', help='name of neural network module')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[64, 64], help='size of patches to sample from Dataset elements')
    parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing Dataset')
    parser.add_argument('--path_dataset_val_csv', type=str, help='path to csv for constructing validation Dataset (evaluated everytime the model is saved)')
    parser.add_argument('--path_run_dir', default='saved_models', help='base directory for saved models')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--shuffle_images', action='store_true', help='set to shuffle images in BufferedPatchDataset')
    parser.add_argument('--transform_signal', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset signal')
    parser.add_argument('--transform_target', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset target')
    parser.add_argument('--scale_signal', default=None, help='scikit-leran scaler for signal')
    parser.add_argument('--scale_target', default=None, help='scikit-leran scaler for target')
    parser.add_argument('--read_channel_from', type=int, default=15, help='start reading signal from channel')
    parser.add_argument('--early_stopping', action='store_true', help='set to perform early stopping')
    parser.add_argument('--final_chan', type=int, default=1, help='number of target parameters')
    parser.add_argument('--propper_kwargs', type=json.loads, default={}, help='path to output directory')
    parser.add_argument('--path_spatial_model', type=str, help='path spatially pretrained model')
    opts = parser.parse_args()
    opts.nn_kwargs['final_chan'] = opts.final_chan
    
    time_start = time.time()
    if not os.path.exists(opts.path_run_dir):
        os.makedirs(opts.path_run_dir)
    if len(opts.iter_checkpoint) > 0:
        path_checkpoint_dir = os.path.join(opts.path_run_dir, 'checkpoints')
        if not os.path.exists(path_checkpoint_dir):
            os.makedirs(path_checkpoint_dir)

    #Setup logging
    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(opts.path_run_dir, 'run.log'), mode='a')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)

    #Set random seed
    if opts.seed is not None:
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)

    #Instantiate Model
    path_model = os.path.join(opts.path_run_dir, 'model.p')
    if os.path.exists(path_model):
        model = fnet.load_model_from_dir(opts.path_run_dir, gpu_ids=opts.gpu_ids)
        logger.info('model loaded from: {:s}'.format(path_model))
    elif (opts.path_spatial_model is not None) and not (os.path.exists(path_model)):
        print(f'Loading spatial model from: {opts.path_spatial_model}')
        basic_model = fnet.fnet_model.Model(
            nn_module=opts.nn_module,
            lr=opts.lr,
            gpu_ids=opts.gpu_ids,
            nn_kwargs=opts.nn_kwargs,
        )
        model = fnet.load_spatial_model(basic_model, opts.path_spatial_model, gpu_ids=opts.gpu_ids)
        print(model)
        logger.info('spatial model loaded from: {:s}'.format(opts.path_spatial_model))
    else:
        model = fnet.fnet_model.Model(
            nn_module=opts.nn_module,
            lr=opts.lr,
            gpu_ids=opts.gpu_ids,
            nn_kwargs=opts.nn_kwargs,
        )
        logger.info('Model instianted from: {:s}'.format(opts.nn_module))
    logger.info(model)

    # Freeze model parameters of the spatial part
    if opts.path_spatial_model is not None:
        model.net.net_recurse.requires_grad_(False)

    #Load saved history if it already exists
    path_losses_csv = os.path.join(opts.path_run_dir, 'losses.csv')
    if os.path.exists(path_losses_csv):
        fnetlogger = fnet.FnetLogger(path_losses_csv)
        logger.info('History loaded from: {:s}'.format(path_losses_csv))
    else:
        fnetlogger = fnet.FnetLogger(columns=['num_iter', 'loss_batch'])

    #Get scaler for signal and targets
    path_scaler = os.path.join(opts.path_run_dir, 'scaler', 'train','')
    if not os.path.exists(path_scaler):
        os.makedirs(path_scaler)
    if opts.scale_signal is not None:
        make_scaler(opts.path_dataset_csv, path_scaler, scaler_type=opts.scale_signal, col='signal', start=opts.read_channel_from)
    if opts.scale_target: 
        make_scaler(opts.path_dataset_csv, path_scaler, scaler_type=opts.scale_target, col='target')
    if opts.path_dataset_val_csv is not None:
        path_scaler_val = os.path.join(opts.path_run_dir, 'scaler', 'train','')
        if not os.path.exists(path_scaler_val):
            os.makedirs(path_scaler_val)
        if opts.scale_signal:
            make_scaler(opts.path_dataset_val_csv, path_scaler_val, scaler_type=opts.scale_signal, col='signal', start=opts.read_channel_from)
        if opts.scale_target: 
            make_scaler(opts.path_dataset_val_csv, path_scaler_val, scaler_type=opts.scale_target, col='target')
    
    # # get propper to crop images
    # img = envi.open(dir_r+fname+prefix_r[:-4]+'.hdr')
    # bands = np.asarray(img.metadata['wavelength']).astype(float)
    # propper = fnet.transforms.Propper(**opts.propper_kwargs)

    #Get data loader; def. loss criterion; load history, if exists
    n_remaining_iterations = max(0, (opts.n_iter - model.count_iter))
    dataloader_train = get_dataloader(n_remaining_iterations, opts)
    if opts.path_dataset_val_csv is not None:
        dataloader_val = get_dataloader(n_remaining_iterations, opts, validation=True)
        criterion_val = model.criterion_fn()
        path_losses_val_csv = os.path.join(opts.path_run_dir, 'losses_val.csv')
        if os.path.exists(path_losses_val_csv):
            fnetlogger_val = fnet.FnetLogger(path_losses_val_csv)
            logger.info('History loaded from: {:s}'.format(path_losses_val_csv))
        else:
            fnetlogger_val = fnet.FnetLogger(columns=['num_iter', 'loss_val'])
    
    with open(os.path.join(opts.path_run_dir, 'train_options.json'), 'w') as fo:
        json.dump(vars(opts), fo, indent=4, sort_keys=True)

    elapsed_time = 0
    for i, (signal, target) in enumerate(dataloader_train, model.count_iter):
        tic =time.time()
        loss_batch = model.do_train_iter(signal, target)
        fnetlogger.add({'num_iter': i + 1, 'loss_batch': loss_batch})
        print('num_iter: {:6d} | loss_batch: {:.4f}'.format(i + 1, loss_batch))
        dict_iter = dict(
            num_iter = i + 1,
            loss_batch = loss_batch,
        )
        # model.scheduler.step() # LR scheduler

        if ((i + 1) % opts.interval_save == 0) or ((i + 1) == opts.n_iter):
            if opts.path_dataset_val_csv is not None:
                loss_val_sum = 0
                for idx_val, (signal_val, target_val) in enumerate(dataloader_val): 
                    pred_val = model.predict(signal_val)
                    if model.net.final_chan == 1:
                        loss_val_batch = criterion_val(pred_val, target_val.squeeze()).item()
                    else:
                        loss_val_batch = criterion_val(pred_val, target_val).item()
                    loss_val_sum += loss_val_batch
                    print('loss_val_batch: {:.4f}'.format(loss_val_batch))
                loss_val = loss_val_sum/len(dataloader_val)
                print('loss_val: {:.4f}'.format(loss_val))
                fnetlogger_val.add({'num_iter': i + 1, 'loss_val': loss_val})
                fnetlogger_val.to_csv(path_losses_val_csv)
                logger.info('loss val log saved to: {:s}'.format(path_losses_val_csv))
            if opts.early_stopping and fnetlogger_val.data['num_iter'][-1] > opts.interval_save:
                if fnetlogger_val.data['loss_val'][-2] < loss_val:
                    print('Stopping early, last saved model at iteration {}'.format(fnetlogger_val.data['num_iter'][-2]))
                    model.save_state
                    break
            model.save_state(path_model)
            fnetlogger.to_csv(path_losses_csv)
            logger.info('BufferedPatchDataset buffer history: {}'.format(dataloader_train.dataset.get_buffer_history()))
            logger.info('loss log saved to: {:s}'.format(path_losses_csv))
            logger.info('model saved to: {:s}'.format(path_model))
            logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))
        if (i + 1) in opts.iter_checkpoint:
            path_save_checkpoint = os.path.join(path_checkpoint_dir, 'model_{:06d}.p'.format(i + 1))
            model.save_state(path_save_checkpoint)
            logger.info('model checkpoint saved to: {:s}'.format(path_save_checkpoint))
        toc=time.time()
        elapsed_time += (toc-tic)
        print(f"elapsed time: {elapsed_time} s")

if __name__ == '__main__':
    main()
    