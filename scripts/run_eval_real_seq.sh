#!/bin/bash

EXE_PATH='/scratch/fs1/stewal/repos/Gated2Depth/src'
INTERPRETER='/scratch/fs1/stewal/Software/miniconda3/envs/tf1.9-gpu/bin/python'
split="test" # or "val"
data_dir="/media/stewal/Steffi/selfsupervised_g2d_results/example_sequences/depth_estimation"
results_dir="/media/stewal/Steffi/selfsupervised_g2d_results/example_sequences/depth_estimation"
gpu=1

model_dir="/scratch/fs1/stewal/repos/Gated2Depth/models/gated2depth_real_day/model.ckpt-13460"
#model_dir="/scratch/fs1/stewal/repos/Gated2Depth/models/gated2depth_real_night/model.ckpt-8028"
eval_files="/media/stewal/Steffi/selfsupervised_g2d_results/splits/example_sequences/2018-10-29_16-18-47.txt"
$INTERPRETER "$EXE_PATH/train_eval.py" \
    --results_dir $results_dir \
	  --model_dir $model_dir \
	  --eval_files_path $eval_files \
	  --base_dir $data_dir \
	  --data_type real \
	  --gpu $gpu \
	  --mode eval \
	  --dataset "g2d"
