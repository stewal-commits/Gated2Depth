#!/bin/bash

split="test" # or "val"
data_dir="/media/stewal/a03d8631-ecbb-412f-bb84-b3f36c6f85a5/jbod-dense/fs1/datasets/iccv2019/real/depth_estimation"
model_dir="./models/gated2depth_real_day/model.ckpt-13460"
results_dir="./results/gated2depth_real_day/${split}"
eval_files="./splits/g2d_stamps/real_${split}_day.txt"
gpu=0

python src/train_eval.py \
    --results_dir $results_dir \
	  --model_dir $model_dir \
	  --eval_files_path $eval_files \
	  --base_dir $data_dir \
	  --data_type real \
	  --gpu $gpu \
	  --mode eval \
	  --binned_metric \
	  --dataset "g2d"




