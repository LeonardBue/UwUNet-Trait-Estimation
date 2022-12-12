#!/bin/bash -x
# estimate traits from HSI
# to run file from bash command line:
# ./code/uwu_net/scripts/predict_UwU_real.sh <DATASET> 0

DATASET=${1:-LMmed}
MODEL_DIR="saved_models/${DATASET}"
N_IMAGES=1000
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

TEST_OR_TRAIN=test
python predict.py \
       --class_dataset RasterDataset \
       --final_chan 4 \
       --transform_signal fnet.transforms.do_nothing \
       --transform_target fnet.transforms.do_nothing \
       --path_model_dir ${MODEL_DIR} \
       --path_dataset_csv "../../data/Hyperion/${DATASET}/${TEST_OR_TRAIN}.csv"\
       --n_images ${N_IMAGES} \
       --no_signal \
       --no_prediction_unpropped \
       --path_save_dir "../../data/Hyperion/${DATASET}/results/${TEST_OR_TRAIN}" \
       --gpu_ids ${GPU_IDS} \
       --scale_target MinMaxScaler \
       --no_target \
       --read_channel_from 0 \
       --merge_output \

# add "/${TEST_OR_TRAIN}" to path_run_dir if different data set is used

TEST_OR_TRAIN=train
python predict.py \
       --class_dataset RasterDataset \
       --final_chan 4 \
       --transform_signal fnet.transforms.do_nothing \
       --transform_target fnet.transforms.do_nothing \
       --path_model_dir ${MODEL_DIR} \
       --path_dataset_csv "../../data/Hyperion/${DATASET}/${TEST_OR_TRAIN}.csv"\
       --n_images ${N_IMAGES} \
       --no_signal \
       --no_prediction_unpropped \
       --path_save_dir "../../data/Hyperion/${DATASET}/results/${TEST_OR_TRAIN}" \
       --gpu_ids ${GPU_IDS} \
       --scale_target MinMaxScaler \
       --no_target \
       --read_channel_from 0 \
       --merge_output \
