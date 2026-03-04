model='histgen'
max_length=100
epochs=40
region_size=96
prototype_num=512

python /home/woody/iwi5/iwi5204h/HistGen/main_test_AllinOne.py \
    --image_dir /home/woody/iwi5/iwi5204h/HistGen/Data/WSI/PatchResult/pt_files/dinov2_vitl/ \
    --ann_path /home/woody/iwi5/iwi5204h/HistGen/Data/Label/train_val_test.json \
    --dataset_name wsi_report \
    --model_name $model \
    --max_seq_length $max_length \
    --threshold 1 \
    --batch_size 1 \
    --epochs $epochs \
    --step_size 10 \
    --topk 512 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --region_size $region_size \
    --prototype_num $prototype_num \
    --save_dir /home/woody/iwi5/iwi5204h/HistGen/Data/TestResult_OG/17_seed46/Best_16/ \
    --gamma 0.8 \
    --seed 46 \
    --log_period 1000 \
    --load /home/woody/iwi5/iwi5204h/HistGen/Data/TrainingResult_17_seed46/model_best_epoch16.pth \
    --beam_size 3