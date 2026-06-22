"""
main.py – Full pipeline runner
Usage:
    python main.py
    python main.py --skip-train
    python main.py --notes 200 --temperature 0.8
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def banner(step, title):
    print("\n" + "="*60)
    print(f"  STEP {step}: {title}")
    print("="*60)

def run_pipeline(skip_train=False, num_notes=150, temperature=0.7):
    banner(1, "Collect MIDI Data")
    from collect_midi_data import collect_data
    collect_data(n_classical=10, n_jazz=5, n_blues=5)

    banner(2, "Preprocess MIDI → Sequences")
    from preprocess_data import preprocess
    preprocess()

    if not skip_train:
        banner(3, "Build & Train LSTM Model (PyTorch)")
        from train_model import train
        train()
    else:
        print("\n⏭️   Skipping training – using saved weights.\n")

    banner(5, "Generate New Music")
    from generate_music import generate
    generate(num_notes=num_notes, temperature=temperature)

    print("\n" + "="*60)
    print("  🎵  PIPELINE COMPLETE")
    print("="*60)
    print("\n  📁  output/midi/   ← open .mid in Windows Media Player")
    print("  📁  models/        ← weights + training_curves.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train',  action='store_true')
    parser.add_argument('--notes',       type=int,   default=150)
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()
    run_pipeline(args.skip_train, args.notes, args.temperature)
