#!/bin/bash

split="test"
data_dir="/external/10g/dense2/fs1/datasets/202210_GatedStereoDatasetv3"
dataset="gatedstereo"
results_dir="/external/10g/dense2/fs1/datasets/202210_GatedStereoDatasetv3"

gpu=1

daytime="day"
model_dir="/external/10g/dense2/fs1/students/stewal/models/Gated2Depth/models/gated2depth_real_day/model.ckpt-13460"

eval_files="./splits/gatedstereo/test_${daytime}_gatedstereo.txt"
python src/train_eval.py \
    --results_dir $results_dir \
    --model_dir $model_dir \
    --eval_files_path $eval_files \
    --base_dir $data_dir \
    --data_type real \
    --gpu $gpu \
    --dataset $dataset \
    --mode eval \
    --min_distance 3. \
    --max_distance 160.


daytime="night"
model_dir="/external/10g/dense2/fs1/students/stewal/models/Gated2Depth/models/gated2depth_real_night/model.ckpt-8028"
eval_files="./splits/gatedstereo/test_${daytime}_gatedstereo.txt"
python src/train_eval.py \
    --results_dir $results_dir \
    --model_dir $model_dir \
    --eval_files_path $eval_files \
    --base_dir $data_dir \
    --data_type real \
    --gpu $gpu \
    --dataset $dataset \
    --mode eval \
    --min_distance 3. \
    --max_distance 160.


