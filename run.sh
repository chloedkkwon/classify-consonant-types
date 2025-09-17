#!/bin/bash
# run.sh - Complete ML pipeline for HuBERT Korean stop consonant classification

set -e  # Exit on any error

# Default mode
MODE="train"
CUSTOM_MODEL=""
SAVE_RESULTS=false
RESULTS_FILE="test_results.csv"
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    train|test|both)
      MODE="$1"
      shift
      ;;
    --model|-m)
      CUSTOM_MODEL="$2"
      shift 2
      ;;
    --save-results)
      SAVE_RESULTS=true
      shift
      ;;
    --results-file)
      RESULTS_FILE="$2"
      shift 2
      ;;
    --help|-h)
      HELP=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

if [ "$HELP" = true ]; then
  echo "Usage: $0 [MODE] [OPTIONS]"
  echo ""
  echo "MODES:"
  echo "  train          Train a new model (default)"
  echo "  test           Test an existing model"
  echo "  both           Train and then test (train + test)"
  echo ""
  echo "OPTIONS:"
  echo "  --model, -m PATH    Use specific model checkpoint for testing"
  echo "  --save-results         Save inference results to CSV file"
  echo "  --results-file FILE    Output filename for results (default: test_results.csv)"
  echo "  --help, -h          Show this help message"
  echo ""
  echo "Environment variables (override defaults):"
  echo "  CSV_FILE       Path to CSV file (default: data/T_all.csv)"
  echo "  AUDIO_DIR      Path to audio directory (default: data/audio)"
  echo "  OUTPUT_DIR     Model output directory (default: model)"
  echo "  BATCH_SIZE     Batch size (default: 8)"
  echo "  FREEZE_EPOCHS  Freeze epochs (default: 3)"
  echo "  FT_EPOCHS      Fine-tuning epochs (default: 5)"
  echo "  LR_HEAD        Head learning rate (default: 2e-4)"
  echo "  LR_ALL         Full model learning rate (default: 1e-5)"
  echo "  SPLIT          Train/test split ratio (default: 0.8)"
  echo ""
  echo "Examples:"
  echo "  $0                              # Train a new model"
  echo "  $0 train                        # Train a new model (explicit)"
  echo "  $0 test                         # Test with model/best.pt"

  echo "  $0 test --model custom.pt       # Test with specific model"
  echo "  $0 both                         # Train then test"
  echo "  $0 test --save-results          # Test and save results to CSV"
  echo "  $0 test --model custom.pt --save-results --results-file my_results.csv"
  echo ""
  echo "  CSV_FILE=data/new.csv $0 train  # Train with different dataset"
  echo "  BATCH_SIZE=16 $0 both           # Use larger batch size"
  exit 0
fi

# Configuration (can be overridden by environment variables)
CSV_FILE="${CSV_FILE:-data/T_all.csv}"
AUDIO_DIR="${AUDIO_DIR:-data/audio}"
OUTPUT_DIR="${OUTPUT_DIR:-model}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-8}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-3}"
FT_EPOCHS="${FT_EPOCHS:-5}"
LR_HEAD="${LR_HEAD:-2e-4}"
LR_ALL="${LR_ALL:-1e-5}"
TUNE_MODE="${TUNE_MODE:-two_phase}"
SPLIT="${SPLIT:-0.8}"

# Determine checkpoint path
if [ -n "$CUSTOM_MODEL" ]; then
  CKPT_PATH="$CUSTOM_MODEL"
else
  CKPT_PATH="${CKPT_PATH:-$OUTPUT_DIR/best.pt}"
fi

# Validation
if [ ! -f "$CSV_FILE" ]; then
  echo "CSV file not found: $CSV_FILE"
  exit 1
fi

if [ ! -d "$AUDIO_DIR" ]; then
  echo "Audio directory not found: $AUDIO_DIR"
  exit 1
fi

# Create output directory for training modes
if [ "$MODE" = "train" ] || [ "$MODE" = "both" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Add results saving arguments for test modes
RESULTS_ARGS=()
if [ "$SAVE_RESULTS" = true ]; then
  RESULTS_ARGS+=(--save-results --results-file "$RESULTS_FILE")
fi

# Common arguments for all Python calls
COMMON_ARGS=(
  --csv "$CSV_FILE"
  --audio-dir "$AUDIO_DIR"
  --batch-size "$BATCH_SIZE"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
  --ckpt-path "$CKPT_PATH"
  --fp16
)

# Training-specific arguments
TRAIN_ARGS=(
  --freeze-epochs "$FREEZE_EPOCHS"
  --ft-epochs "$FT_EPOCHS"
  --lr-head "$LR_HEAD"
  --lr-all "$LR_ALL"
  --tune-mode "$TUNE_MODE"
)

echo "HuBERT Korean Stop Consonant Classification Pipeline"
echo "  Dataset: $CSV_FILE"
echo "  Audio: $AUDIO_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Checkpoint: $CKPT_PATH"
echo "  Batch size: $BATCH_SIZE, Split: $SPLIT"
echo ""

case $MODE in
  "train")
    echo "Mode: Training only"
    echo "  Config: $TUNE_MODE (freeze=$FREEZE_EPOCHS, ft=$FT_EPOCHS)"
    echo "  Learning rates: head=$LR_HEAD, full=$LR_ALL"
    echo ""
    
    python run.py --mode train "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}"
    echo "✅ Training complete! Best model saved to $CKPT_PATH"
    ;;
    
  "test")
    echo "Mode: Testing only"
    echo "  Model: $CKPT_PATH"
    echo ""
    
    if [ ! -f "$CKPT_PATH" ]; then
      echo "  Checkpoint not found at $CKPT_PATH"
      echo ""
      echo "  Available checkpoints in $OUTPUT_DIR:"
      find "$OUTPUT_DIR" -name "*.pt" -type f 2>/dev/null | head -10 || echo "   No checkpoints found"
      echo ""
      exit 1
    fi
    
    python run.py --mode test "${COMMON_ARGS[@]}" "${RESULTS_ARGS[@]}"
    echo "✅ Testing complete!"
    
    # Show model info
    echo ""
    echo "Model Summary:"
    echo "   Checkpoint: $CKPT_PATH"
    echo "   Size: $(du -h "$CKPT_PATH" | cut -f1)"
    echo "   Modified: $(stat -f "%Sm" "$CKPT_PATH" 2>/dev/null || stat -c "%y" "$CKPT_PATH" 2>/dev/null || echo "Unknown")"
    ;;
    
  "both")
    echo "Mode: Training + Testing"
    echo "  Config: $TUNE_MODE (freeze=$FREEZE_EPOCHS, ft=$FT_EPOCHS)"
    echo "  Learning rates: head=$LR_HEAD, full=$LR_ALL"
    echo ""
    
    python run.py --mode train_and_test "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}" "${RESULTS_ARGS[@]}"
    echo "✅ Training and testing complete!"
    echo "  Best model saved to $CKPT_PATH"
    ;;
    
  *)
    echo "Unknown mode: $MODE"
    echo "Valid modes: train, test, both"
    echo "Use --help for more information"
    exit 1
    ;;
esac

echo ""
echo "Pipeline finished successfully!"