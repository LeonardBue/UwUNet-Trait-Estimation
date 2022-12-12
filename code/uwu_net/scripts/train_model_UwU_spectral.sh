#!/bin/bash -x
# train spectrally for trait estimation
# set -i (and --no_shuffle) to train combined model
# to run file from bash command line:
# ./code/uwu_net/scripts/train_model_UwU_synthetic.sh <DATASET> 0

DATASET=${1:-LMmed}
BUFFER_SIZE=1 # must be 1 for individual models
N_ITER=2500 #22560 
RUN_DIR="saved_models/inform_${DATASET}"
PATH_DATASET_ALL_CSV="../../data/dbs/inform_${DATASET}.csv"
PATH_DATASET_TRAIN_CSV="../../data/dbs/inform_${DATASET}/train.csv" 
PATH_DATASET_TEST_CSV="../../data/dbs/inform_${DATASET}/test.csv"
PATH_SPATIAL_MODEL="saved_models/${DATASET}"
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..
pwd

python scripts/python/split_dataset.py ${PATH_DATASET_ALL_CSV} "../../data/dbs/" -v #-i #--no_shuffle #  
python train_model.py \
       --nn_module fnet_nn_UwU \
       --final_chan 4 \
       --n_iter ${N_ITER} \
       --lr 0.004 \
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --path_dataset_val_csv ${PATH_DATASET_TEST_CSV} \
       --class_dataset RasterDataset \
       --transform_signal fnet.transforms.do_nothing  \
       --transform_target fnet.transforms.do_nothing \
       --patch_size 64 64\
       --batch_size 8 \
       --buffer_size ${BUFFER_SIZE} \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS} \
       --scale_target MinMaxScaler \
       --buffer_switch_frequency 64001 \
       --path_spatial_model ${PATH_SPATIAL_MODEL} \
      
