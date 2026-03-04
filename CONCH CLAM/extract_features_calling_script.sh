#!/bin/bash
models="conch_v1_5"
declare -A gpus
gpus["conch_v1_5"]=0

for model in $models
do
        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}
        python3 /home/woody/iwi5/iwi5204h/CLAM/extract_features_fp.py \
                --data_h5_dir "/home/woody/iwi5/iwi5204h/HistGen/Data/WSI/PatchResult" \
                --data_slide_dir "/home/janus/iwi5-datasets/REG2025/Train_01/" \
                --slide_ext ".tiff" \
                --csv_path "/home/woody/iwi5/iwi5204h/HistGen/Data/Label/output.csv" \
                --feat_dir "/home/woody/iwi5/iwi5204h/CLAM/CONCH_1_5_features" \
                --model_name "conch_v1_5" \
                --batch_size 512 \
                --target_patch_size 448

done