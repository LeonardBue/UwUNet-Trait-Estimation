#!/bin/bash -x
# train spatial for spatial reconstruction
# set -i (and --no_shuffle) to train combined model
# to run file from bash command line:
# ./code/uwu_net/scripts/train_model_UwU_spectral.sh <DATASET> 0

DATASET=${1:-LMmed}
BUFFER_SIZE=1
N_ITER=22560 
RUN_DIR="saved_models/${DATASET}"
PATH_DATASET_ALL_CSV="../../data/Hyperion/${DATASET}.csv"
PATH_DATASET_TRAIN_CSV="../../data/Hyperion/${DATASET}/train.csv" 
PATH_DATASET_TEST_CSV="../../data/Hyperion/${DATASET}/test.csv"
PATH_SPATIAL_MODEL="saved_models/${DATASET}"
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..
pwd

python scripts/python/split_dataset.py ${PATH_DATASET_ALL_CSV} "../../data/Hyperion/" -v --train_size 0.8 #-i #--no_shuffle
python train_model.py \
       --nn_module fnet_nn_UwU \
       --final_chan 97 \
       --n_iter ${N_ITER} \
       --lr 0.0001 \
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --path_dataset_val_csv ${PATH_DATASET_TEST_CSV} \
       --class_dataset RasterDataset \
       --transform_signal fnet.transforms.do_nothing  \
       --transform_target fnet.transforms.do_nothing \
       --patch_size 64 64\
       --batch_size 2 \
       --buffer_size ${BUFFER_SIZE} \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS} \
       --read_channel_from 0 \
       --buffer_switch_frequency 2 \
       --shuffle_images \
       

       
