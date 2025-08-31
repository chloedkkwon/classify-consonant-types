import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torchaudio
import os

label_dict = {"plain": 0, "tense": 1, "aspirated": 2}

class KoreanStopDataset(Dataset):
    def __init__(self, dataframe, audio_dir):
        self.data = dataframe
        self.audio_dir = audio_dir
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row["fname"] + ".wav")
        waveform, sr = torchaudio.load(wav_path)

        if sr != 16000:
            waveform = self.resampler(waveform)

        label = label_dict[row["cons_type"]]  # <-- update
        return waveform.squeeze(), torch.tensor(label)

def collate_fn(batch):
    """
    Pads input waveforms.
    Args:
        batch: List of (waveform_tensor, label)
    Returns:
        input_values: Padded audio tensor [batch, max_len]
        labels: Tensor of shape [batch]
    """
    waveforms, labels = zip(*batch)
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0) 
    # print("Padded:", padded_waveforms.shape)
    
    # Reshape the padded waveforms to have shape [batch_size, 1, max_length]
    # 1 is the number of channels (since we're working with raw waveforms)
    # padded_waveforms = padded_waveforms.unsqueeze(1)  # Adding the channels dimension
    # print("Padded-squeezed:", padded_waveforms.shape)

    # input = model(padded_waveforms) 

    # inputs = processor(padded_waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
    # input_values = inputs.input_values  # [batch, max_length]
    # print("Input shape:", input_values.shape)

    labels = torch.tensor(labels)

    return padded_waveforms, labels
