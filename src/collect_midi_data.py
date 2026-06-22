"""
STEP 1: collect_midi_data.py
Download or generate sample MIDI data for training.
Place your own .mid files in data/midi_files/ OR run this script
to auto-generate sample classical-style MIDI files for demo/testing.
"""

import os
import random
from midiutil import MIDIFile

# ── Output folder ──────────────────────────────────────────────
MIDI_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'midi_files')
os.makedirs(MIDI_DIR, exist_ok=True)

# ── Musical building blocks ────────────────────────────────────
MAJOR_SCALE   = [0, 2, 4, 5, 7, 9, 11]          # W W H W W W H
MINOR_SCALE   = [0, 2, 3, 5, 7, 8, 10]           # W H W W H W W
BLUES_SCALE   = [0, 3, 5, 6, 7, 10]

CHORD_PROGRESSIONS = [
    [0, 5, 3, 4],   # I–VI–IV–V  (pop/classical)
    [0, 3, 4, 0],   # I–IV–V–I   (blues)
    [0, 5, 1, 4],   # I–VI–II–V  (jazz ii-V-I variant)
    [0, 2, 5, 4],   # I–III–VI–V
]

RHYTHMS = [
    [1, 1, 1, 1],           # quarter notes
    [0.5, 0.5, 1, 1, 1],   # two eighths + quarters
    [2, 1, 1],              # half + two quarters
    [0.5, 0.5, 0.5, 0.5, 1, 1],  # lots of eighths
]


def make_scale(root: int, scale_intervals: list, octaves: int = 2) -> list:
    notes = []
    for o in range(octaves):
        notes += [root + o * 12 + i for i in scale_intervals]
    return notes


def generate_melody(scale_notes: list, num_notes: int = 32) -> list:
    melody = []
    prev = random.choice(scale_notes)
    for _ in range(num_notes):
        # Prefer stepwise motion but allow occasional leaps
        step = random.choice([-2, -1, -1, 0, 1, 1, 2, 3, -3])
        idx  = scale_notes.index(prev) if prev in scale_notes else 0
        idx  = max(0, min(len(scale_notes) - 1, idx + step))
        prev = scale_notes[idx]
        melody.append(prev)
    return melody


def generate_midi_file(filename: str, style: str = 'classical',
                        tempo: int = 120, num_bars: int = 16):
    """Generate a single MIDI file with melody + simple accompaniment."""
    midi = MIDIFile(2)          # 2 tracks: melody + bass

    # ── Track 0 – Melody ──────────────────────────────────────
    midi.addTempo(0, 0, tempo)
    midi.addTrackName(0, 0, f"{style.title()} Melody")

    if style == 'jazz':
        root, scale_intervals = 60, BLUES_SCALE
        prog = random.choice(CHORD_PROGRESSIONS[2:])
    elif style == 'blues':
        root, scale_intervals = 57, BLUES_SCALE
        prog = CHORD_PROGRESSIONS[1]
    else:                        # classical / default
        root, scale_intervals = 60, MAJOR_SCALE
        prog = CHORD_PROGRESSIONS[0]

    scale_notes = make_scale(root, scale_intervals)
    melody      = generate_melody(scale_notes, num_bars * 4)
    rhythm      = random.choice(RHYTHMS)

    time = 0.0
    for i, pitch in enumerate(melody):
        dur = rhythm[i % len(rhythm)]
        vel = random.randint(60, 100)
        midi.addNote(0, 0, pitch, time, dur, vel)
        time += dur

    # ── Track 1 – Bass / Accompaniment ────────────────────────
    midi.addTempo(1, 0, tempo)
    midi.addTrackName(1, 0, f"{style.title()} Bass")

    time = 0.0
    for bar in range(num_bars):
        chord_root = root - 12 + prog[bar % len(prog)] * 2
        for beat in range(4):
            midi.addNote(1, 0, chord_root,      time, 0.5, 70)
            midi.addNote(1, 0, chord_root + 7,  time + 0.5, 0.5, 60)
            time += 1

    # ── Save ──────────────────────────────────────────────────
    filepath = os.path.join(MIDI_DIR, filename)
    with open(filepath, 'wb') as f:
        midi.writeFile(f)
    print(f"  ✅  Saved: {filepath}")
    return filepath


def collect_data(n_classical: int = 10, n_jazz: int = 5, n_blues: int = 5):
    """Generate a small training dataset of MIDI files."""
    print("\n🎵  Generating MIDI training data …\n")
    files = []

    for i in range(n_classical):
        f = generate_midi_file(
            f"classical_{i+1:02d}.mid", style='classical',
            tempo=random.randint(80, 130), num_bars=16)
        files.append(f)

    for i in range(n_jazz):
        f = generate_midi_file(
            f"jazz_{i+1:02d}.mid", style='jazz',
            tempo=random.randint(100, 160), num_bars=12)
        files.append(f)

    for i in range(n_blues):
        f = generate_midi_file(
            f"blues_{i+1:02d}.mid", style='blues',
            tempo=random.randint(70, 110), num_bars=12)
        files.append(f)

    print(f"\n📂  Total files generated: {len(files)}")
    print(f"📁  Location : {os.path.abspath(MIDI_DIR)}")
    print("\n💡  TIP: You can also drop your own .mid files into")
    print(f"         {os.path.abspath(MIDI_DIR)}")
    return files


if __name__ == '__main__':
    collect_data()
