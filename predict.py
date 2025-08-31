# python predict.py --wav sample.wav --model_path model/saved_model.pt


import torch
import torchaudio
import argparse
from src.hubert_classifier import KoreanStopClassifier
from src.utils import load_model
import os

LABELS = ["plain", "tense", "aspirated"]

def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # [T]

def predict(model, waveform, device):
    model.eval()
    waveform = waveform.unsqueeze(0).to(device)  # [1, T]
    with torch.no_grad():
        logits = model(waveform)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
    return LABELS[pred], probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Path to .wav file or folder")
    parser.add_argument("--model_path", type=str, default="model/saved_model.pt")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "attention"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KoreanStopClassifier(pooling=args.pooling)
    model = load_model(model, args.model_path, device)

    if os.path.isdir(args.wav):
        files = [os.path.join(args.wav, f) for f in os.listdir(args.wav) if f.endswith(".wav")]
    else:
        files = [args.wav]

    for path in files:
        waveform = load_audio(path)
        pred_label, probs = predict(model, waveform, device)
        print(f"{os.path.basename(path)} → Predicted: {pred_label} | Probabilities: {probs.round(3)}")

if __name__ == "__main__":
    main()
