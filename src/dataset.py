import torch, torchaudio
from torch.utils.data import Dataset
import os

class KoreanStopDataset(Dataset):
    def __init__(self, df, audio_dir, label_col="cons_type", label2id=None, target_sr=16000):
        """
        df: pnadas DataFrame with columns ["fname", label_col]
        audio_dir: directory where audio files are stored
        label_col: column name for labels
        label2id: dictionary mapping label strings to integer ids
        """
        self.df = df
        self.audio_dir = audio_dir
        self.label_col = label_col
        self.label2id = label2id or {l: i for i, l in enumerate(sorted(df[label_col].unique()))}
        self.target_sr = target_sr

        self.file_col = "path" if "path" in self.df.columns else "fname"

    def __len__(self):
        return len(self.df)
    
    def _load_resample_mono(self, fullpath):
        wav, sr = torchaudio.load(fullpath) # (C, T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # convert to mono by averaging channels
        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)
        return wav.squeeze(0)  # return as 1D tensor (T,)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row["path"] if "path" in row else os.path.join(self.audio_dir, row["fname"] + ".wav")
        waveform = self._load_resample_mono(wav_path)
        label = self.label2id[row[self.label_col]]
        return {"input_values": waveform, "labels": label}
