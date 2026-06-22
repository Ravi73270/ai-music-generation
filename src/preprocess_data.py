"""
STEP 2: preprocess_data.py
Parse MIDI files → extract note/chord tokens → encode → build sequences.
Saves X.npy (N, seq_len, 1), y.npy (N,) int labels, vocab pickles.
"""

import os, pickle
import numpy as np
from collections import Counter
from music21 import converter, instrument, note, chord

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
MIDI_DIR = os.path.join(BASE_DIR, 'data', 'midi_files')
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEQ_LEN  = 50


def _extract_notes(filepath: str) -> list:
    notes_out = []
    try:
        midi_stream = converter.parse(filepath)
        parts = instrument.partitionByInstrument(midi_stream)
        elements = parts.parts[0].recurse() if parts else midi_stream.flat.notes
        for el in elements:
            if isinstance(el, note.Note):
                notes_out.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                token = '.'.join(str(int(p.ps)) for p in sorted(el.pitches, key=lambda p: p.ps))
                notes_out.append(token)
    except Exception as e:
        print(f"  ⚠️  Skipping {os.path.basename(filepath)}: {e}")
    return notes_out


def parse_midi_files(midi_dir: str) -> list:
    files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
    if not files:
        raise FileNotFoundError(f"No .mid files in {midi_dir}. Run collect_midi_data.py first.")
    print(f"\n📂  Found {len(files)} MIDI file(s)\n")
    all_notes = []
    for fname in sorted(files):
        notes = _extract_notes(os.path.join(midi_dir, fname))
        all_notes.extend(notes)
        print(f"  ✅  {fname:30s} → {len(notes):4d} tokens")
    print(f"\n🎵  Total: {len(all_notes)} tokens  |  Unique: {len(set(all_notes))}")
    return all_notes


def build_vocab(notes: list):
    vocab     = sorted(set(notes))
    token2int = {t: i for i, t in enumerate(vocab)}
    int2token = {i: t for i, t in enumerate(vocab)}
    print(f"📖  Vocab size: {len(vocab)}")
    return token2int, int2token


def build_sequences(notes: list, token2int: dict):
    encoded = [token2int[t] for t in notes]
    X_raw, y_raw = [], []
    for i in range(len(encoded) - SEQ_LEN):
        X_raw.append(encoded[i:i + SEQ_LEN])
        y_raw.append(encoded[i + SEQ_LEN])

    n_vocab = len(token2int)
    # Normalise X to [0,1]; y stays as int labels (CrossEntropyLoss needs indices)
    X = np.reshape(X_raw, (len(X_raw), SEQ_LEN, 1)).astype(np.float32) / float(n_vocab)
    y = np.array(y_raw, dtype=np.int64)

    print(f"🔢  Patterns: {len(X_raw)}  |  X:{X.shape}  y:{y.shape}")
    return X, y


def preprocess():
    notes            = parse_midi_files(MIDI_DIR)
    top10 = Counter(notes).most_common(10)
    print(f"\n🏆  Top tokens: {[t for t,_ in top10]}")
    token2int, int2token = build_vocab(notes)
    X, y             = build_sequences(notes, token2int)

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, 'X.npy'), X)
    np.save(os.path.join(DATA_DIR, 'y.npy'), y)
    for name, obj in [('notes', notes), ('token2int', token2int), ('int2token', int2token)]:
        with open(os.path.join(DATA_DIR, f'{name}.pkl'), 'wb') as f:
            pickle.dump(obj, f)

    print(f"\n💾  Saved to {os.path.abspath(DATA_DIR)}/")
    print("🎉  Preprocessing complete!\n")
    return X, y, token2int, int2token, notes


if __name__ == '__main__':
    preprocess()
