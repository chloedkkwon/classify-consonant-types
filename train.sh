#!/bin/bash
# train.sh

set -e

# Parse command line arguments
SKIP_TEST=false
TEST_ONLY=false
HELP=false

while [[ $# -gt 0 ]]; do
  case $1 in
  --skip-test)
    SKIP_TEST=true
    shift
    ;;
  --TEST_ONLY=true
    shift
    ;;
  --help|-h)
    HELP=true
    shift
    ;;
  *)
    echo "Unknown option: $1"
    echo "Use -- help for usage information"
    exit 1
    ;;
  esac
done

if [ "$HELP" = true ]; then
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "OPTIONS:"
  echo "  --skip-test    Train only, skip testing phase"
  echo "  --test-only    Skip training, only run testing"
  echo "  --help, -h     Show this help message"
  echo ""
  echo "Environment variables (override defaults):"
  echo "  CSV_FILE       Path to CSV file (default: data/T_all.csv)"
  echo "  AUDIO_DIR      Path to audio directory (default: data/audio)"
  echo "  OUTPUT_DIR     Model output directory (default: model)"
  echo "  CKPT_PATH      Checkpoint path (default: \$OUTPUT_DIR/best.pt)"
  echo "  BATCH_SIZE     Batch size (default: 8)"
  echo "  FREEZE_EPOCHS  Freeze epochs (default: 3)"
  echo "  FT_EPOCHS      Fine-tuning epochs (default: 5)"
  exit 0
fi

# Configuration
CSV_FILE="${CSV_FILE:-data/T_all.csv}"
AUDIO_DIR="${AUDIO_DIR:-data/audio}"
OUTPUT_DIR="${OUTPUT_DIR:-model}"
CKPT_PATH="${CKPT_PATH:-$OUTPUT_DIR/best.pt}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-8}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-3}"
FT_EPOCHS="${FT_EPOCHS:-5}"
LR_HEAD="${LR_HEAD:-2e-4}"
LR_ALL="${LR_ALL:-1e-5}"
TUNE_MODE="${TUNE_MODE:-two_phase}"
SPLIT="${SPLIT:-0.8}"

# Validation
if [ ! -f "$CSV_FILE" ]; then
  echo "CSV file not found: $CSV_FILE"
  exit 1
fi

if [ ! -d "$AUDIO_DIR" ]; then
  echo "Audio directory not found: $AUDIO_DIR"
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Common arguments
COMMON_ARGS=(
  --csv "$CSV_FILE"
  --audio_dir "$AUDIO_DIR"
  --batch_size "$BATCH_SIZE"
  --split "$SPLIT"
  --output_dir "$OUTPUT_DIR"
  --ckpt_path "$CKPT_PATH"
  --fp16
)

# Training arguments
TRAIN_ARGS=(
  --freeze_epochs "$FREEZE_EPOCHS"
  --ft_epochs "$FT_EPOCHS"
  --lr_head "$LR_HEAD"
  --lr_all "$LR_ALL"
  --tune_mode "$TUNE_MODE"
)

echo "Starting training pipeline..."
echo "  CSV: $CSV_FILE"
echo "  Audio dir: $AUDIO_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Checkpoint: $CKPT_PATH"
echo "  Mode: $TUNE_MODE (freeze=$FREEZE_EPOCHS, ft=$FT_EPOCHS)"
echo "  Batch size: $BATCH_SIZE, Split: $SPLIT"

if [ "$TEST_ONLY" = true ]; then
  echo "Running test-only mode..."
  
  if [ ! -f "$CKPT_PATH" ]; then
    echo "Checkpoint not found at $CKPT_PATH"
    echo "Train a model first or specify a valid checkpoint path"
    exit 1
  fi
  
  python run.py --mode test "${COMMON_ARGS[@]}"
  
elif [ "$SKIP_TEST" = true ]; then
  echo " Training only (skipping test)..."
  
  python run.py --mode train "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}"
  echo "✅ Training complete! Best model saved to $CKPT_PATH"
  
else
  echo "Training + Testing..."
  
  python run.py --mode train_and_test "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}"
  echo "✅ Training and testing complete! Best model at $CKPT_PATH"
fi

echo "Pipeline finished successfully!"