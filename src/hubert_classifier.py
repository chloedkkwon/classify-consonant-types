import torch
import torch.nn as nn
from transformers import HubertModel
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HUGGINGFACE_HUB_TOKEN in your environment (as an .evn file)")

# If you use huggingface_hub login:
# from huggingface_hub import login
# login(token=HF_TOKEN)

class AttentionPooling(nn.Module):
    # Applies soft attention pooling over the time dimension
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states):
        # hidden_states: [batch, time, hidden_dim]
        attn_weights = self.attn(hidden_states).squeeze(-1)  # [batch, time]
        attn_weights = torch.softmax(attn_weights, dim=1)    # soft attention weights
        pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden_dim]
        return pooled

class KoreanStopClassifier(nn.Module):
    def __init__(self, model_name="team-lucid/hubert-large-korean", 
                 token = HF_TOKEN,
                 pooling="mean", freeze_encoder=True):
        """
        Korean stop classifier using HuBERT and optional attention pooling.

        Args:
            model_name (str): HuggingFace model name.
            pooling (str): "mean" or "attention".
            freeze_encoder (bool): Whether to freeze the HuBERT encoder.
        """
        super().__init__()
        self.encoder = HubertModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size  # typically 768

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if pooling == "mean":
            self.pooling = lambda x: x.mean(dim=1)
        elif pooling == "attention":
            self.pooling = AttentionPooling(self.hidden_dim)
        else:
            raise ValueError("pooling must be 'mean' or 'attention'")

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)  # output: plain, tense, aspirated
        )

    def forward(self, input_values):
        """
        Args:
            input_values (Tensor): shape [B, T] — raw waveform.
        Returns:
            logits (Tensor): shape [B, 3]
        """
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.encoder.parameters())):
            outputs = self.encoder(input_values)
            hidden_states = outputs.last_hidden_state  # [batch, time, hidden_dim]

        pooled = self.pooling(hidden_states)  # [batch, hidden_dim]
        logits = self.classifier(pooled)      # [batch, 3]
        return logits

