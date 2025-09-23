#!/bin/bash

# Base directories
BASE_DIR="results"
DATA_DIR="/data/aimix/Spain/Barcelona/blindsweeps_fMP4"  # Directory containing fold-specific data files

# Create experiment directory
# mkdir -p "$BASE_DIR"

# Training configuration
MAX_EPOCHS=200
EVAL_FREQ=10
PATIENCE=10

# Function to start training for a specific fold
train_fold() {
    local fold=$1
    local gpu_id=$2
    
    # Create fold-specific directories
    local fold_dir="$BASE_DIR/fold_${fold}"
    mkdir -p "$fold_dir"
    
    # Construct data file paths based on fold
    # local train_file="$DATA_DIR/fold_$((fold+1))_finetune_train.csv"
    local train_file="/home/tanya-akumu/gestation_age/dataset/data/all_opt_train_split.csv"
    local val_file="/home/tanya-akumu/gestation_age/dataset/data/all_opt_val_split.csv" # "$DATA_DIR/fold_$((fold+1))_finetune_val.csv"
    local model="usfm_noatt" # "resnet"
    local sampling="disim" # "optimal"
    # Start training in background
    python train.py \
        --fold $fold \
        --gpu_id $gpu_id \
        --train_file $train_file \
        --val_file $val_file \
        --save_dir $fold_dir \
        --sampling $sampling \
        --model $model \
        --max_epochs $MAX_EPOCHS \
        --eval_frequency $EVAL_FREQ \
        --patience $PATIENCE \
        > "$fold_dir/$model-$sampling-stdout.log" 2> "$fold_dir/$model-$sampling-stderr.log" &
    
    echo "Started training $model with $sampling sampling for fold $fold on GPU $gpu_id"
}

# Start training all folds in parallel
# Assuming you have 5 GPUs (0-4), one for each fold
for fold in 0
do
    train_fold $fold 3
done

# Wait for all background processes to complete
wait

echo "All folds completed training"
