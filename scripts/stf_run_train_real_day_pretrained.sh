#!/bin/bash

daytime="day" # or "day"
data_dir="/media/stewal/Datasets/Algolux_allv3"
model_dir="./models/gated2depth_real_${daytime}"
results_dir="/external/models/models/Gated2Depth/real_${daytime}_pretrained"
disc_path='./src/exported_disc/disc.pb'
train_files="./splits/stf/train_${daytime}.txt"
eval_files="./splits/stf/val_${daytime}.txt"
dataset='stf'
gpu=0

python src/train_eval.py \
    --results_dir $results_dir \
    --model_dir $model_dir \
    --train_files_path $train_files \
    --eval_files_path $eval_files \
    --base_dir $data_dir \
    --data_type real \
    --gpu $gpu \
    --dataset $dataset \
    --mode train \
    --num_epochs 1 \
    --exported_disc_path $disc_path \
    --lrate 0.0001 \
    --smooth_weight 1.0\
    --adv_weight 1.0\
    --use_multiscale\
    --use_filtered_lidar
