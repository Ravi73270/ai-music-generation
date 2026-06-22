"""
STEP 5: generate_music.py
Load trained PyTorch LSTM → generate note sequences → save MIDI.

Usage:
    python src/generate_music.py
    python src/generate_music.py --notes 200 --temperature 0.8
"""

import os, sys, argparse, pickle, random
import numpy as np
import torch
from datetime import datetime
from music21 import stream, note, chord, instrument, tempo

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model

BASE_DIR  = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUT_MIDI  = os.path.join(BASE_DIR, 'output', 'midi')
OUT_AUDIO = os.path.join(BASE_DIR, 'output', 'audio')
os.makedirs(OUT_MIDI,  exist_ok=True)
os.makedirs(OUT_AUDIO, exist_ok=True)
SEQ_LEN = 50


def load_vocab():
    with open(os.path.join(DATA_DIR, 'notes.pkl'),     'rb') as f: notes     = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'token2int.pkl'), 'rb') as f: token2int = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'int2token.pkl'), 'rb') as f: int2token = pickle.load(f)
    print(f"📖  Vocab: {len(token2int)} tokens")
    return notes, token2int, int2token


def load_model(n_vocab: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model(n_vocab=n_vocab, seq_len=SEQ_LEN).to(device)

    best   = os.path.join(MODEL_DIR, 'music_lstm_best.pt')
    final  = os.path.join(MODEL_DIR, 'music_lstm_final.pt')
    if os.path.exists(best):
        model.load_state_dict(torch.load(best, map_location=device))
        print(f"✅  Loaded: {best}")
    elif os.path.exists(final):
        model.load_state_dict(torch.load(final, map_location=device))
        print(f"✅  Loaded: {final}")
    else:
        print("⚠️  No trained weights found → demo mode (random output).")
        print("    Run  src/train_model.py  first for real music.\n")

    model.eval()
    return model, device


def sample_temperature(logits: torch.Tensor, temperature: float) -> int:
    logits = logits / max(temperature, 1e-8)
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()
    return int(np.random.choice(len(probs), p=probs))


def generate_sequence(model, notes, token2int, int2token, n_vocab,
                      num_notes=150, temperature=0.7, device='cpu'):
    print(f"\n🎼  Generating {num_notes} notes  (temp={temperature}) …")
    start    = random.randint(0, len(notes) - SEQ_LEN - 1)
    seed     = [token2int[t] for t in notes[start: start + SEQ_LEN]]
    generated = []

    with torch.no_grad():
        for step in range(num_notes):
            x = torch.tensor(seed, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(-1) / float(n_vocab)  # (1, seq, 1)
            x = x.to(device)
            logits, _ = model(x)
            idx = sample_temperature(logits[0], temperature)
            generated.append(int2token[idx])
            seed.append(idx)
            seed = seed[1:]
            if (step + 1) % 50 == 0:
                print(f"  … {step+1}/{num_notes}")

    print(f"✅  Generated {len(generated)} tokens")
    return generated


def tokens_to_stream(tokens: list) -> stream.Stream:
    output_notes = []
    offset = 0.0
    for token in tokens:
        if '.' in token:
            parts = token.split('.')
            chord_notes = []
            for p in parts:
                try:    chord_notes.append(note.Note(int(p)))
                except: pass
            if chord_notes:
                c = chord.Chord(chord_notes)
                c.offset = offset
                output_notes.append(c)
        else:
            try:
                n = note.Note(token)
            except:
                try:    n = note.Note(int(token))
                except: continue
            n.offset = offset
            n.storedInstrument = instrument.Piano()
            output_notes.append(n)
        offset += 0.5
    s = stream.Stream(output_notes)
    s.insert(0, tempo.MetronomeMark(number=120))
    return s


def save_note_sheet(tokens, filename):
    path = os.path.join(OUT_AUDIO, filename)
    with open(path, 'w') as f:
        f.write("AI-Generated Music – Note Sheet\n" + "="*40 + "\n\n")
        for i, t in enumerate(tokens, 1):
            f.write(f"  {i:4d}.  {t}\n")
        f.write("\n\n-- HOW TO CONVERT MIDI TO AUDIO --\n")
        f.write("1. Online:  https://signal.vercel.app  (drag & drop .mid)\n")
        f.write("2. Free DAW: LMMS (Windows) – open .mid and export .wav\n")
        f.write("3. CLI: fluidsynth -ni soundfont.sf2 output.mid -F output.wav\n")
    print(f"📄  Note sheet → {path}")


def generate(num_notes=150, temperature=0.7):
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    notes, token2int, int2token = load_vocab()
    n_vocab   = len(token2int)
    model, device = load_model(n_vocab)
    tokens    = generate_sequence(model, notes, token2int, int2token,
                                  n_vocab, num_notes, temperature, device)
    midi_s    = tokens_to_stream(tokens)
    midi_path = os.path.join(OUT_MIDI, f"generated_{ts}.mid")
    midi_s.write('midi', fp=midi_path)
    print(f"💾  MIDI → {midi_path}")
    save_note_sheet(tokens, f"note_sheet_{ts}.txt")
    print(f"\n🎉  Done!  Open output/midi/generated_{ts}.mid in any MIDI player.")
    return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notes',       type=int,   default=150)
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()
    generate(args.notes, args.temperature)
