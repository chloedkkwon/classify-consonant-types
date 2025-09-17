# HuBERT Korean Consonant Classification

A deep learning project for classifying Korean stop consonant types (aspirated, tense, plain) using HuBERT (Hidden-Unit BERT) transformer models.

## Project Overview

This project fine-tunes a pre-trained HuBERT model to classify Korean stop consonants into three categories:
- **Aspirated** (격음): ㅋ, ㅌ, ㅍ, ㅊ
- **Tense** (경음): ㄲ, ㄸ, ㅃ, ㅆ, ㅉ  
- **Plain** (평음): ㄱ, ㄷ, ㅂ, ㅅ, ㅈ

The model uses the `team-lucid/hubert-large-korean` pre-trained model and implements a two-phase fine-tuning approach for optimal performance.

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training)
- Audio files in supported formats (WAV, MP3, FLAC)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/chloedkkwon/classify-consonant-types.git
cd classify-consonant-types
```

2. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate classifyenv
```

3. **Prepare your data:**
   - Place your CSV file in `data/` directory
   - Place audio files in `data/audio/` directory
   - See `data/README.md` for format requirements

### Usage

The project provides a simple bash interface through `run.sh`:

#### Training Only
```bash
./run.sh train
```

#### Testing Only (requires pre-trained model)
```bash
./run.sh test --save-results
```

#### Train and Test
```bash
./run.sh both --save-results
```

#### Using Custom Parameters
```bash
# Use custom dataset
CSV_FILE=data/my_data.csv ./run.sh train

# Adjust training parameters  
BATCH_SIZE=16 FT_EPOCHS=10 ./run.sh train

# Test with specific model
./run.sh test --model model/my_model.pt --save-results
```

### Advanced Usage (Python Script)

For more control, use the Python script directly:

```bash
# Training
python run.py --mode train \
    --csv data/T_all.csv \
    --audio-dir data/audio \
    --batch-size 8 \
    --ft-epochs 5

# Testing with results
python run.py --mode test \
    --csv data/T_all.csv \
    --audio-dir data/audio \
    --ckpt-path model/best.pt \
    --save-results \
    --results-file results/detailed_results.csv
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CSV_FILE` | `data/T_all.csv` | Path to dataset CSV |
| `AUDIO_DIR` | `data/audio` | Audio files directory |
| `OUTPUT_DIR` | `model` | Model checkpoint directory |
| `BATCH_SIZE` | `8` | Training batch size |
| `FREEZE_EPOCHS` | `3` | Epochs with frozen backbone |
| `FT_EPOCHS` | `5` | Fine-tuning epochs |
| `LR_HEAD` | `2e-4` | Learning rate for classification head |
| `LR_ALL` | `1e-5` | Learning rate for full model |
| `SPLIT` | `0.8` | Train/validation split ratio |

### Model Parameters

- **Base Model**: `team-lucid/hubert-large-korean`
- **Audio Processing**: 16kHz sampling rate, 3-second max length
- **Training Strategy**: Two-phase (freeze backbone → fine-tune all)
- **Architecture**: HuBERT + classification head

## Results

After training, the model will be saved to `model/best.pt`. Test results include:

- **Accuracy metrics** on test set
- **Per-class probabilities** for each prediction
- **Confidence scores** for model predictions
- **Detailed CSV output** with all predictions

Example test output:
```
[Test] loss 0.2341 acc 89.567
✅ Testing complete!
Saved results to test_results.csv
```

## 🐳 Docker Support

For containerized deployment and reproducible environments:

```bash
# Build container
docker build -t hubert-korean-classifier .

# Run training
docker run --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/model:/app/model \
  hubert-korean-classifier \
  conda run -n classifyenv ./run.sh train

# Run testing
docker run --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/model:/app/model:ro \
  -v $(pwd)/results:/app/results \
  hubert-korean-classifier \
  conda run -n classifyenv ./run.sh test --save-results
```

See `Dockerfile` and `docker-compose.yml` for complete containerization setup.

## Data Format

### CSV Requirements

Your dataset CSV should contain:
- `cons_type` or `label`: Target labels (aspirated/tense/plain)
- `filename` or `audio_path`: Path to audio files
- Audio files should be in `data/audio/` directory

Example CSV:
```csv
filename,cons_type
audio_001.wav,aspirated
audio_002.wav,tense
audio_003.wav,plain
```

### Audio Requirements

- **Formats**: WAV, MP3, FLAC
- **Sampling Rate**: Any (automatically resampled to 16kHz)
- **Length**: Any (automatically trimmed/padded to 3 seconds)
- **Channels**: Mono or stereo (converted to mono)

## Development

### Project Components

- **`run.py`**: Main script with argument parsing and training/testing logic
- **`run.sh`**: User-friendly bash interface with presets
- **`src/hubert_classifier_new.py`**: Model architecture and training functions
- **`src/dataset.py`**: Dataset loading and preprocessing
- **`environment.yml`**: Complete dependency specification
