#!/bin/bash

export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760

source_dir="/home/janus/iwi5-datasets/REG2025/Train_01"
wsi_format="tiff"
patch_size=512
save_dir="/home/woody/iwi5/iwi5204h/HistGen/Data/WSI/PatchResult"
preset_file="/home/woody/iwi5/iwi5204h/HistGen/CLAM/presets/tcga.csv"
file_list="/home/woody/iwi5/iwi5204h/HistGen/CLAM/patching_scripts/missing_files.txt"

echo "=== PROCESSING MULTIPLE FILES FROM LIST ==="
echo "Source directory: $source_dir"
echo "File list: $file_list"
echo "Save directory: $save_dir"
echo "==========================================="

# Check if file list exists
if [ ! -f "$file_list" ]; then
    echo "ERROR: File list $file_list not found!"
    exit 1
fi

# Show file list contents
echo "File list contents:"
echo "-------------------"
cat "$file_list"
echo "-------------------"

# Create temporary directory for safety
temp_dir="/home/woody/iwi5/iwi5204h/temp_wsi_batch"
mkdir -p "$temp_dir"

# Create symbolic links for all files in the list
linked_count=0
skipped_count=0
while IFS= read -r wsi_file; do
    # Skip empty lines and comments
    if [[ -z "$wsi_file" || "$wsi_file" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Remove any trailing/leading whitespace
    wsi_file=$(echo "$wsi_file" | xargs)
    
    # Construct full path
    full_path="${source_dir}/${wsi_file}"
    
    # Check if file exists
    if [ ! -f "$full_path" ]; then
        echo "SKIPPED: File not found: $full_path"
        skipped_count=$((skipped_count + 1))
        continue
    fi
    
    # Create symbolic link without moving any data
    ln -s "$full_path" "$temp_dir/$wsi_file"
    echo "LINKED: $wsi_file"
    linked_count=$((linked_count + 1))
    
done < "$file_list"

echo "========================================="
echo "Files linked: $linked_count"
echo "Files skipped: $skipped_count"
echo "========================================="

if [ $linked_count -eq 0 ]; then
    echo "ERROR: No files were linked!"
    rmdir "$temp_dir"
    exit 1
fi

# Verify link functionality
echo "Verifying symbolic links..."
ls -la "$temp_dir"

# Run CLAM processing on the temporary directory
echo "Starting CLAM processing..."
echo "Processing $linked_count files..."

python /home/woody/iwi5/iwi5204h/HistGen/CLAM/create_patches_sg.py \
    --source "$temp_dir" \
    --save_dir "$save_dir" \
    --preset "$preset_file" \
    --patch_level 0 \
    --patch_size "$patch_size" \
    --step_size "$patch_size" \
    --wsi_format "$wsi_format" \
    --patch \
    --seg \
    --stitch

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo "CLAM processing completed successfully!"
    
    # Show results
    echo "Results saved to: $save_dir"
    echo "Generated files:"
    ls -la "$save_dir"
else
    echo "CLAM processing failed!"
fi

# Clean up temporary directory ONLY
echo "Cleaning up temporary directory..."
rm -rf "$temp_dir"

echo "Cleanup complete - original data is safe!"
echo "========================================="