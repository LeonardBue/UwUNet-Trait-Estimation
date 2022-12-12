import argparse
import fnet.data
import fnet.fnet_model
import json
import joblib
import numpy as np
import optuna
import os
import seaborn as sns
import sys
import time
import torch

sns.set()
sys.path.append('../../code/')
from utils import make_scaler
from train_model import get_dataloader


def objective(trial):
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser()
    factor_yx = 1.0  # 0.108 um/px -> 0.29 um/px changed to 1.0 12_03_2019
    default_resizer_str = 'fnet.transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
    parser.add_argument('--batch_size', type=int, default=4, help='size of each batch')
    parser.add_argument('--bpds_kwargs', type=json.loads, default={}, help='kwargs to be passed to BufferedPatchDataset')
    parser.add_argument('--buffer_size', type=int, default=5, help='number of images to cache in memory')
    parser.add_argument('--buffer_switch_frequency', type=int, default=720, help='BufferedPatchDataset buffer switch frequency')
    parser.add_argument('--class_dataset', default='CziDataset', help='Dataset class')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
    parser.add_argument('--interval_save', type=int, default=1000, help='iterations between saving log/model')
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
    parser.add_argument('--transform_target', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset target'),
    parser.add_argument('--scale_signal', default=None, help='scikit-leran scaler for signal'),
    parser.add_argument('--scale_target', default=None, help='scikit-leran scaler for target'),
    parser.add_argument('--read_channel_from', type=int, default=15, help='start reading signal from channel')
    parser.add_argument('--final_chan', type=int, default=1, help='number of target parameters')
    opts = parser.parse_args()
    opts.nn_kwargs['final_chan'] = opts.final_chan
    
    # if not os.path.exists(opts.path_run_dir):
    #     os.makedirs(opts.path_run_dir)
    # if len(opts.iter_checkpoint) > 0:
    #     path_checkpoint_dir = os.path.join(opts.path_run_dir, 'checkpoints')
    #     if not os.path.exists(path_checkpoint_dir):
    #         os.makedirs(path_checkpoint_dir)

    #Set random seed
    if opts.seed is not None:
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)
    opts.path_run_dir = 'saved_models/optuna'

    # Optuna parameters
    lr = trial.suggest_float('lr', 0.0001, 0.005, log=True)
    # p = trial.suggest_float('p', 0.0, 0.5, step=0.1)
    # momentum = trial.suggest_float('momentum', 0.0, 0.9, step=0.01)
    # mult_chan = trial.suggest_int('mult chan', 2, 32, step=10)
    intermediate_chan = trial.suggest_int('intermediate chan', 36, 84, step=12)
    depth = 0# trial.suggest_int('depth', 0, 4)
    # kernel_size = trial.suggest_int('kernel size', 1, 7, step=2)
    # patch_size = trial.suggest_categorical('patch size', [16, 32, 64])
    # patch_size = 32
    # n_patches = (64 // patch_size) * (128 // patch_size)
    # batch_size = trial.suggest_int('batch size', 2, n_patches)
    beta_1 = trial.suggest_float('beta_1', 0.5, 0.9, step=0.1)
    beta_2 = trial.suggest_float('beta_2', 0.9, 0.999, log=True)
    
    # opts.patch_size = [patch_size]*2
    # opts.batch_size = batch_size
    opts.nn_kwargs['intermediate_chan'] = intermediate_chan
    opts.nn_kwargs['depth'] = depth# , 'p': p, 'kernel_size': kernel_size}

    #Instantiate Model
    model = fnet.fnet_model.Model(
        nn_module=opts.nn_module,
        lr=lr, # opts.lr,
        gpu_ids=opts.gpu_ids,
        nn_kwargs=opts.nn_kwargs,
        betas = (beta_1, beta_2)
    )

    #Get scaler for signal and targets
    path_scaler = os.path.join(opts.path_run_dir, 'scaler', 'train','')
    if not os.path.exists(path_scaler):
        os.makedirs(path_scaler)
    if opts.scale_signal is not None:
        make_scaler(opts.path_dataset_csv, path_scaler, scaler_type=opts.scale_signal, col='signal', start=opts.read_channel_from)
    if opts.scale_target: 
        make_scaler(opts.path_dataset_csv, path_scaler, scaler_type=opts.scale_target, col='target')
    if opts.path_dataset_val_csv is not None:
        path_scaler_val = os.path.join(opts.path_run_dir, 'scaler', 'test','')
        if not os.path.exists(path_scaler_val):
            os.makedirs(path_scaler_val)
        if opts.scale_signal:
            make_scaler(opts.path_dataset_val_csv, path_scaler_val, scaler_type=opts.scale_signal, col='signal', start=opts.read_channel_from)
        if opts.scale_target: 
            make_scaler(opts.path_dataset_val_csv, path_scaler_val, scaler_type=opts.scale_target, col='target')

    #Get data loader; def. loss criterion; load history, if exists
    n_remaining_iterations = max(0, (opts.n_iter - model.count_iter))
    dataloader_train = get_dataloader(n_remaining_iterations, opts)
    if opts.path_dataset_val_csv is not None:
        dataloader_val = get_dataloader(n_remaining_iterations, opts, validation=True)
        criterion_val = model.criterion_fn()

    elapsed_time = 0
    loss = 0.
    for i, (signal, target) in enumerate(dataloader_train, model.count_iter):
        tic =time.time()
        loss_batch = model.do_train_iter(signal, target)
        loss += loss_batch
        print(f'iteration: {i+1} \t loss_batch: {loss_batch:.4f}')

        if ((i + 1) % opts.interval_save == 0) or ((i + 1) == opts.n_iter):
            if opts.path_dataset_val_csv is not None:
                loss_val_sum = 0
                for idx_val, (signal_val, target_val) in enumerate(dataloader_val): 
                    pred_val = model.predict(signal_val)

                    loss_val_batch = criterion_val(pred_val, target_val).item()
                    loss_val_sum += loss_val_batch
                    print('loss_val_batch: {:.3f}'.format(loss_val_batch))
                loss_val = loss_val_sum/len(dataloader_val)
                print('loss_val: {:.3f}'.format(loss_val))
        
        toc=time.time()
        elapsed_time += (toc-tic)
        print(f'elapsed time: {elapsed_time}')
    
    return loss_val #loss_batch #loss/opts.n_iter

def print_best_callback(study, trial):
    print(f'Current best value: {study.best_value} with params: {study.best_trial.params}')


# %%
def main():
    path_study = 'saved_models/optuna/study.pkl'
    if os.path.exists(path_study):
        print(f'------------------ loading studi from path: {path_study} -----------------')
        study = joblib.load(path_study)
    else:
        study = optuna.create_study(sampler=optuna.samplers.QMCSampler())
    study.optimize(objective, n_trials=100, callbacks=[print_best_callback])
    joblib.dump(study, path_study)

    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    fig1 = optuna.visualization.plot_parallel_coordinate(study)
    fig2 = optuna.visualization.plot_param_importances(study)
    fig1.write_image('../../draft/graphics/optuna_parallel_coordinate.svg')
    fig2.write_image('../../draft/graphics/optuna_param_importances.svg')

if __name__ == '__main__':
    main()
    
# %%
