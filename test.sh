#!/bin/bash

# test.sh

# Set default values
CSV_FILE="data/T_all.csv"
AUDIO_DIR="data/audio"
MODE="test"
BATCH_SIZE=8
MODEL_PATH="model/saved_model.pt"

echo "Running test script..."

python run.py \
  --mode $MODE \
  --csv $CSV_FILE \
  --audio_dir $AUDIO_DIR \
  --batch_size $BATCH_SIZE \
  --save_path $MODEL_PATH

echo "Testing complete!"
