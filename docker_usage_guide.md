# Docker Usage Guide for HuBERT Korean Consonant Classification

## Project Structure
```
classify_consonant_types/
├── data/
│   ├── T_all.csv
│   └── audio/
├── model/
├── src/
│   ├── __init__.py
│   ├── hubert_classifier_new.py
│   └── dataset.py
├── run.py
├── run.sh
├── Dockerfile
├── Dockerfile.cpu
├── docker-compose.yml
├── environment.yml
├── environment-cpu.yml
└── .dockerignore
```

## Building the Container

### GPU Version (for training):
```bash
docker build -t hubert-korean-classifier:latest .
```

### CPU Version (for inference):
```bash
docker build -f Dockerfile.cpu -t hubert-korean-classifier:cpu .
```

### Using Docker Compose:
```bash
# Build and start GPU version
docker-compose up -d

# Build and start CPU version
docker-compose --profile cpu up -d
```

## Running the Container

### 1. Interactive Shell (Recommended for Development):
```bash
# GPU version
docker run -it --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/results:/app/results \
  hubert-korean-classifier:latest

# CPU version
docker run -it \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/model:/app/model:ro \
  -v $(pwd)/results:/app/results \
  hubert-korean-classifier:cpu
```

### 2. Direct Training:
```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/model:/app/model \
  hubert-korean-classifier:latest \
  conda run -n classifyenv ./run.sh train
```

### 3. Direct Testing:
```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/model:/app/model:ro \
  -v $(pwd)/results:/app/results \
  hubert-korean-classifier:latest \
  conda run -n classifyenv ./run.sh test --save-results --results-file /app/results/test_results.csv
```

### 4. Training + Testing:
```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/results:/app/results \
  hubert-korean-classifier:latest \
  conda run -n classifyenv ./run.sh both --save-results
```

## Using Docker Compose (Easier)

### Start container:
```bash
docker-compose up -d
```

### Execute commands:
```bash
# Training
docker-compose exec hubert-classifier conda run -n classifyenv ./run.sh train

# Testing
docker-compose exec hubert-classifier conda run -n classifyenv ./run.sh test --save-results

# Both
docker-compose exec hubert-classifier conda run -n classifyenv ./run.sh both --save-results

# Interactive shell (conda environment already activated)
docker-compose exec hubert-classifier bash
```

### Stop container:
```bash
docker-compose down
```

## Environment Variables

You can override defaults by setting environment variables:

```bash
docker run --gpus all \
  -e CSV_FILE=/app/data/custom.csv \
  -e AUDIO_DIR=/app/data/custom_audio \
  -e BATCH_SIZE=16 \
  -e FT_EPOCHS=10 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/model:/app/model \
  hubert-korean-classifier:latest \
  conda run -n classifyenv ./run.sh train
```

## Volume Mounts Explained

- `./data:/app/data:ro` - Your data directory (read-only)
- `./model:/app/model` - Model checkpoint storage (read-write)
- `./results:/app/results` - Output results directory (read-write)

## GPU Requirements

For GPU support, you need:
1. NVIDIA Docker runtime installed
2. `--gpus all` flag or equivalent Docker Compose configuration

## Conda Environment Notes

- The container uses your exact conda environment (`classifyenv`)
- All commands should be prefixed with `conda run -n classifyenv`
- The environment is automatically activated in interactive sessions

## Troubleshooting

### Check container logs:
```bash
docker-compose logs hubert-classifier
```

### Debug inside container:
```bash
docker-compose exec hubert-classifier bash
conda activate classifyenv
python -c "import torch; print(torch.cuda.is_available())"
```

### Check mounted data:
```bash
docker-compose exec hubert-classifier ls -la /app/data/
docker-compose exec hubert-classifier head /app/data/T_all.csv
```

### Verify conda environment:
```bash
docker-compose exec hubert-classifier conda info --envs
docker-compose exec hubert-classifier conda run -n classifyenv python --version
```
