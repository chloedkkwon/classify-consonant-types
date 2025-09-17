#!/bin/bash

# test.sh

# Set default values
CSV_FILE="data/T_all.csv"
AUDIO_DIR="data/audio"
MODE="test"
BATCH_SIZE=8
SPLIT=0.8
OUTPUT_DIR="model"
CKPT_PATH="$OUTPUT_DIR/best.pt"

echo "Running test..."
echo "CSV: $CSV_FILE"
echo "Audio dir: $AUDIO_DIR"
echo "Checkpoint: $CKPT_PATH"
echo "Split: $SPLIT"

# quick sanity check
if [ ! -f "$CKPT_PATH" ]; then
  echo "❌ Checkpoint not found at $CKPT_PATH"
  exit 1
fi

python run.py \
  --mode test \
  --csv "$CSV_FILE" \
  --audio_dir "$AUDIO_DIR" \
  --batch_size "$BATCH_SIZE" \
  --split "$SPLIT" \
  --output_dir "$OUTPUT_DIR" \
  --ckpt_path "$CKPT_PATH"

echo "Testing complete!"
