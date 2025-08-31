# Classify Consonant Types

Lightweight pipeline for training and evaluating a classifier on Korean consonant categories using audio + metadata. Includes simple CLI, reproducible environment, and examples for training and inference. Files in this repo include `run.py`, `predict.py`, shell helpers (`train.sh`, `test.sh`), a `src/` package, and a conda `environment.yml` to reproduce the setup.

## Features
- **One-command training & inference** via `run.py`
- **Conda environment** for reproducibility (`environment.yml`)
- **Pluggable models** and utilities under `src/`
- **Shell helpers** (`train.sh`, `test.sh`) for common workflows

## Project structure
```
classify-consonant-types/
├─ src/                  # model & data pipeline code
├─ model/                # saved checkpoints (ignored in git; you save here)
├─ run.py                # CLI for training & inference
├─ predict.py            # convenience script for predictions
├─ train.sh, test.sh     # example shell entrypoints
├─ environment.yml       # conda env spec
└─ .gitignore
```

## Installation

### 1) Create the conda environment
```bash
conda env create -f environment.yml
conda activate classifyenv
```

### 2) (Optional) Set up secrets via `.env`
If you use any private services (e.g., Hugging Face), keep tokens out of code:
```bash
# in project root
echo "HUGGINGFACE_HUB_TOKEN=hf_..." >> .env
```

Your code can load it with:
```python
from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
```

## Data format

Training expects a **CSV** pointing to audio files. Minimal columns you’ll likely need:

- `filepath` or `relative_path`: path to each audio file (e.g., `data/audio/xxx.wav`)
- `label`: target consonant class (e.g., `plain`, `tense`, `aspirated`)
- (optional) other metadata you want to use as features or for analysis

> Tip: keep your raw audio under `data/audio/` and **don’t** commit large audio to git. Use `.gitignore` / external storage.

## Quick start

### Train
```bash
python run.py --mode train   --csv data/T_all.csv   --audio_dir data/audio   --epochs 10   --split 0.80
```

This will train a model and save a checkpoint (e.g., `model/classifier.pt`).

### Inference (batch)
```bash
python run.py --mode inference   --csv data/T_all.csv   --audio_dir data/audio   --model_path model/classifier.pt
```

### Inference (single file via predict.py)
```bash
python predict.py --audio path/to/file.wav --model_path model/classifier.pt
```

### Shell helpers
```bash
bash train.sh
bash test.sh
```

## Command-line options

Common flags supported by `run.py`:

- `--mode {train,inference}` – choose workflow
- `--csv PATH` – path to the dataset CSV
- `--audio_dir DIR` – directory containing audio files referenced in CSV
- `--epochs N` – training epochs (train mode)
- `--split FLOAT` – train/valid split ratio (train mode)
- `--model_path PATH` – checkpoint to save (train) or load (inference)

> Run `python run.py -h` to see the full set of arguments and defaults.
