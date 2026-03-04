export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760

source_dir="/home/janus/iwi5-datasets/REG2025/Train_01"
#source_dir="/home/woody/iwi5/iwi5204h/HistGen/Data/WSI"

wsi_format="tiff"

patch_size=512
save_dir="/home/woody/iwi5/iwi5204h/HistGen/Data/WSI/PatchResult"
python /home/woody/iwi5/iwi5204h/HistGen/CLAM/create_patches_sg.py \
    --source $source_dir \
    --save_dir $save_dir\
    --preset /home/woody/iwi5/iwi5204h/HistGen/CLAM/presets/tcga.csv \
    --patch_level 0 \
    --patch_size $patch_size \
    --step_size $patch_size \
    --wsi_format $wsi_format \
    --patch \
    --seg \
    --stitch