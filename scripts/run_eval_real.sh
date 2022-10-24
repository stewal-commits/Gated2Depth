#!/bin/bash

split="test" # or "val"
data_dir="/media/stewal/Steffi/selfsupervised_g2d_results/g2d/depth_estimation"
results_dir="/media/stewal/Steffi/selfsupervised_g2d_results/g2d/depth_estimation"
gpu=0

model_dir="./models/gated2depth_real_day/model.ckpt-13460"
eval_files="/media/stewal/Steffi/selfsupervised_g2d_results/splits/g2d_stamps/real_${split}_day.txt"
python src/train_eval.py \
    --results_dir $results_dir \
	  --model_dir $model_dir \
	  --eval_files_path $eval_files \
	  --base_dir $data_dir \
	  --data_type real \
	  --gpu $gpu \
	  --mode eval \
	  --dataset "g2d"


model_dir="./models/gated2depth_real_night/model.ckpt-8028"
eval_files="/media/stewal/Steffi/selfsupervised_g2d_results/splits/g2d_stamps/real_${split}_night.txt"
python src/train_eval.py \
    --results_dir $results_dir \
	  --model_dir $model_dir \
	  --eval_files_path $eval_files \
	  --base_dir $data_dir \
	  --data_type real \
	  --gpu $gpu \
	  --mode eval \
	  --dataset "g2d"