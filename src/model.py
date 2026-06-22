"""
STEP 3: model.py
Stacked LSTM model using PyTorch (works on Python 3.13 + Windows).

Architecture:
    Input (batch, seq_len, 1)
    → LSTM(512, 3 layers, dropout=0.3)
    → Dense(256, ReLU)
    → Dense(n_vocab, Softmax via CrossEntropyLoss)
"""

import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, n_vocab: int, seq_len: int = 50,
                 hidden_size: int = 512, num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.n_vocab     = n_vocab

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size  = 1,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout,
            batch_first = True
        )
        self.bn1     = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size, 256)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(256, n_vocab)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, 1)
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]           # last time step → (batch, hidden)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)           # raw logits → use CrossEntropyLoss
        return out, hidden

    def init_hidden(self, batch_size: int, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h, c)


def build_model(n_vocab: int, seq_len: int = 50) -> MusicLSTM:
    model = MusicLSTM(n_vocab=n_vocab, seq_len=seq_len)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️  MusicLSTM built  |  vocab={n_vocab}  |  params={total:,}\n")
    print(model)
    return model


if __name__ == '__main__':
    m = build_model(n_vocab=80)
    x = torch.randn(32, 50, 1)
    out, _ = m(x)
    print(f"\n✅  Forward pass OK  |  output shape: {out.shape}")
