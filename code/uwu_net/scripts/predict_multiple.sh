#!/bin/bash
# train_multiple.sh
#
# to run file from bash command line:
# ./code/uwu_net/scripts/predict_multiple.sh

CSV_PATH="D:\Projects\MA\data\Hyperion\images_visual_check.csv"
MODEL="inform_"
pwd

[ ! -f $CSV_PATH ] && { echo "$CSV_PATH file not found"; exit 99; }
{
    read
    echo "Starting to test models."
    while IFS=, read -r index id keep
    do
        # echo $keep
        if [[ $keep == "True"* ]]
        then
            ./code/uwu_net/scripts/predict_UwU.sh "$MODEL$id" "0"
            wait
            echo "Finished predicting with model $MODEL$id."
        else
            echo "skipping id: $id."
        fi
        echo "-----------------------------------------------------"
    done
} < $CSV_PATH
echo "-----------------------------------------------------"
echo "Done with all ids in list!"