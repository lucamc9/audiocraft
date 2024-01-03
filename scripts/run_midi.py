from audiocraft.metrics import ChromaCosineSimilarityMetric
from audiocraft.models import MusicGen
from audiocraft.data.audio_utils import convert_audio, load_melody
from audiocraft.data.audio import audio_write
from tqdm import tqdm
import pandas as pd
import numpy as np
import pathlib
import argparse
import librosa
import torch
import os
import pretty_midi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_path", type=str, required=True, help="path to midi")
    parser.add_argument("--caption", type=str, required=True, help="caption")
    args = parser.parse_args()
    
    sr = 32000
    duration = 10

    # model prep
    model = MusicGen.get_pretrained("facebook/musicgen-melody")
    model.set_generation_params(duration=duration)

    # inference
    melody_midi = pretty_midi.PrettyMIDI(args.midi_path)
    generated = model.generate_with_chroma([args.caption], [melody_midi], sr)
    
    print(f"Generated {generated.shape}")