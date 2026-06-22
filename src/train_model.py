"""
STEP 4: train_model.py
Train the PyTorch LSTM model on preprocessed note sequences.

Usage:  python src/train_model.py

Saves:
    models/music_lstm_best.pt      ← best checkpoint
    models/music_lstm_final.pt     ← final weights
    models/training_curves.png     ← loss/accuracy plot
"""

import os, sys, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 128
LR         = 0.001
SEQ_LEN    = 50


def load_data():
    print("📂  Loading preprocessed data …")
    X = np.load(os.path.join(DATA_DIR, 'X.npy'))      # (N, seq_len, 1)
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))      # (N,) int labels

    with open(os.path.join(DATA_DIR, 'token2int.pkl'), 'rb') as f:
        token2int = pickle.load(f)

    n_vocab = len(token2int)
    print(f"  X:{X.shape}  y:{y.shape}  vocab:{n_vocab}")
    return X, y, n_vocab


def plot_history(losses, accuracies):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses,     color='royalblue');  axes[0].set_title('Training Loss')
    axes[1].plot(accuracies, color='darkorange'); axes[1].set_title('Training Accuracy (%)')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, 'training_curves.png')
    plt.savefig(path, dpi=120)
    print(f"📊  Curves saved → {path}")


def train():
    X_np, y_np, n_vocab = load_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻  Device: {device}")

    # Build tensors
    X_t = torch.tensor(X_np, dtype=torch.float32)   # (N, seq_len, 1)
    y_t = torch.tensor(y_np, dtype=torch.long)       # (N,)

    dataset    = TensorDataset(X_t, y_t)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = build_model(n_vocab=n_vocab, seq_len=SEQ_LEN).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5)

    best_loss = float('inf')
    losses, accuracies = [], []
    patience_counter   = 0
    PATIENCE           = 15

    print(f"\n🚀  Training for up to {EPOCHS} epochs …\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

        avg_loss = epoch_loss / total
        acc      = 100.0 * correct / total
        losses.append(avg_loss)
        accuracies.append(acc)
        scheduler.step(avg_loss)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | {elapsed:.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, 'music_lstm_best.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️   Early stopping at epoch {epoch}")
                break

    # Final save
    torch.save(model.state_dict(),
               os.path.join(MODEL_DIR, 'music_lstm_final.pt'))

    with open(os.path.join(MODEL_DIR, 'training_history.pkl'), 'wb') as f:
        pickle.dump({'loss': losses, 'accuracy': accuracies}, f)

    plot_history(losses, accuracies)
    print(f"\n✅  Training complete!  Best loss: {best_loss:.4f}")
    print(f"    Best   → models/music_lstm_best.pt")
    print(f"    Final  → models/music_lstm_final.pt")
    return model


if __name__ == '__main__':
    train()
