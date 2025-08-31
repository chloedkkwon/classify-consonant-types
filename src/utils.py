# src/utils.py

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os


def collate_fn(batch):
    """Pads variable-length waveforms in batch and returns tensors."""
    waveforms, labels = zip(*batch)
    waveforms_padded = pad_sequence(waveforms, batch_first=True)
    labels_tensor = torch.tensor(labels)
    return waveforms_padded, labels_tensor


def evaluate_model(model, dataloader, criterion, device):
    """Evaluates model on a given dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def save_model(model, path):
    """Saves model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """Loads model weights."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
