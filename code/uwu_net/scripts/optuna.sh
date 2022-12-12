#!/bin/bash -x
# to run file from bash command line:
# ./code/uwu_net/scripts/optuna.sh inform_reflectance 0

DATASET=${1:-LMmed}
BUFFER_SIZE=1 
N_ITER=1000
RUN_DIR="saved_models/optuna_${DATASET}"
PATH_DATASET_ALL_CSV="../../data/dbs/inform_${DATASET}.csv"
PATH_DATASET_TRAIN_CSV="../../data/dbs/inform_${DATASET}/train.csv" 
PATH_DATASET_TEST_CSV="../../data/dbs/inform_${DATASET}/test.csv"
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..
pwd

python scripts/python/split_dataset.py ${PATH_DATASET_ALL_CSV} "../../data/dbs" -v #-i # --train_size 0.8 
python optuna_study.py \
       --nn_module fnet_nn_UwU \
       --n_iter ${N_ITER} \
       --final_chan 4 \
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --path_dataset_val_csv ${PATH_DATASET_TEST_CSV} \
       --class_dataset RasterDataset \
       --transform_signal fnet.transforms.do_nothing  \
       --transform_target fnet.transforms.do_nothing \
       --patch_size 64 64 \
       --batch_size 32 \
       --buffer_size ${BUFFER_SIZE} \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS} \
       --scale_target MinMaxScaler \
       --buffer_switch_frequency 64001

