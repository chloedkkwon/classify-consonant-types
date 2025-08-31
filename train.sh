#!/bin/bash

# train.sh

# Set default values
CSV_FILE="data/T_all.csv"
AUDIO_DIR="data/audio"
MODE="train"
EPOCHS=10
BATCH_SIZE=8
LR=1e-4
SPLIT=0.8
SAVE_PATH="model/model.pt"

echo "Running training script..."

python run.py \
  --mode $MODE \
  --csv $CSV_FILE \
  --audio_dir $AUDIO_DIR \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --split $SPLIT \
  --save_path $SAVE_PATH

echo "Training complete!"
