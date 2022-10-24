#!/bin/bash

split="test"
weathers=( "light_fog" "dense_fog" "clear"  "snow" )
data_dir="/media/stewal/Steffi/selfsupervised_g2d_results/stf/depth_estimation"
dataset="g2d"
results_dir="/media/stewal/Steffi/selfsupervised_g2d_results/stf/depth_estimation"

gpu=1

daytime="day"
model_dir="./models/gated2depth_real_day/model.ckpt-13460"
for weather in "${weathers[@]}"
do
  eval_files="/media/stewal/Steffi/selfsupervised_g2d_results/splits/stf/${split}_${weather}_${daytime}.txt"
  python src/train_eval.py \
      --results_dir $results_dir \
      --model_dir $model_dir \
      --eval_files_path $eval_files \
      --base_dir $data_dir \
      --data_type real \
      --gpu $gpu \
      --dataset $dataset \
      --mode eval
done

daytime="night"
model_dir="./models/gated2depth_real_night/model.ckpt-8028"
for weather in "${weathers[@]}"
do
  eval_files="/media/stewal/Steffi/selfsupervised_g2d_results/splits/stf/${split}_${weather}_${daytime}.txt"
  python src/train_eval.py \
      --results_dir $results_dir \
      --model_dir $model_dir \
      --eval_files_path $eval_files \
      --base_dir $data_dir \
      --data_type real \
      --gpu $gpu \
      --dataset $dataset \
      --mode eval
done

