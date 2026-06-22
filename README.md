# 🎵 AI Music Generation with LSTM
> **CodeAlpha Internship – Task 3**
> By Ravi Sahu | [github.com/Ravi73270](https://github.com/Ravi73270)

---

## 📁 Project Structure

```
ai_music_generation/
│
├── main.py                      ← Run full pipeline (all 5 steps)
│
├── src/
│   ├── collect_midi_data.py     ← STEP 1: Generate/collect MIDI files
│   ├── preprocess_data.py       ← STEP 2: Parse MIDI → note sequences
│   ├── model.py                 ← STEP 3: LSTM model architecture
│   ├── train_model.py           ← STEP 4: Train model + save weights
│   └── generate_music.py        ← STEP 5: Generate new music → MIDI
│
├── data/
│   ├── midi_files/              ← Training MIDI files go here
│   ├── X.npy                    ← Input sequences (auto-generated)
│   ├── y.npy                    ← Target labels (auto-generated)
│   ├── notes.pkl                ← Raw token list
│   ├── token2int.pkl            ← Vocab: token → index
│   └── int2token.pkl            ← Vocab: index → token
│
├── models/
│   ├── music_lstm_best.h5       ← Best checkpoint (ModelCheckpoint)
│   ├── music_lstm_final.h5      ← Final weights
│   └── training_curves.png      ← Loss & accuracy plot
│
├── output/
│   ├── midi/                    ← Generated .mid files
│   └── audio/                   ← Note sheets + conversion instructions
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup (Windows PowerShell)

```powershell
# 1. Navigate to project folder
cd ai_music_generation

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Run the Complete Pipeline

```powershell
# Run all 5 steps at once
python main.py

# Options
python main.py --notes 200           # generate 200 notes
python main.py --temperature 0.9     # more creative output
python main.py --skip-train          # skip training, use saved weights
```

---

## 🔢 Run Steps Individually

```powershell
# Step 1: Generate sample MIDI training data
python src/collect_midi_data.py

# Step 2: Preprocess MIDI → sequences
python src/preprocess_data.py

# Step 3+4: Train LSTM model
python src/train_model.py

# Step 5: Generate music
python src/generate_music.py --notes 150 --temperature 0.7
```

---

## 🎧 Playing the Generated Music

| Method | Steps |
|---|---|
| **Windows** | Double-click the `.mid` file in `output/midi/` → opens in Windows Media Player |
| **Online** | Upload `.mid` to [onlinesequencer.net](https://onlinesequencer.net) or [signal.vercel.app](https://signal.vercel.app) |
| **DAW** | Open in GarageBand (Mac), FL Studio, or LMMS (free, Windows) |
| **CLI** | `fluidsynth -ni soundfont.sf2 output.mid -F output.wav -r 44100` |

---

## 🧠 Model Architecture

```
Input (50, 1)
    → LSTM(512) + BatchNorm + Dropout(0.3)   ← sequence patterns
    → LSTM(512) + BatchNorm + Dropout(0.3)   ← higher-level patterns
    → LSTM(256) + BatchNorm + Dropout(0.3)   ← compressed representation
    → Dense(256, ReLU)
    → Dense(n_vocab, Softmax)                ← next-note prediction
```

- **Loss**: Categorical Cross-Entropy
- **Optimizer**: RMSprop (lr=0.001)
- **Training**: Up to 50 epochs with EarlyStopping + ReduceLROnPlateau

---

## 🎛️ Temperature Parameter

| Value | Effect |
|-------|--------|
| `0.3–0.5` | Very safe, repetitive, stays on familiar patterns |
| `0.7` | Balanced (default) — musical but with some creativity |
| `1.0+` | Chaotic, experimental, unpredictable |

---

## 📊 Tips for Better Results

1. **Add real MIDI files** — drop classical/jazz `.mid` files into `data/midi_files/` for richer training data. Free sources: [musescore.com](https://musescore.com), [kunstderfuge.com](http://www.kunstderfuge.com)
2. **Train longer** — set `EPOCHS = 150` in `train_model.py`
3. **Increase data** — more MIDI files = better model
4. **Try different temperatures** when generating

---

## 📌 Tech Stack

`Python` · `TensorFlow/Keras` · `music21` · `midiutil` · `NumPy` · `Matplotlib`

---
*CodeAlpha AI Internship — June/July 2026*
