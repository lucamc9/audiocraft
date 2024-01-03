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
    melody = pretty_midi.PrettyMIDI(args.midi_path)
    chroma = melody.get_chroma(fs=235/30) # (12, 77)
    chroma = np.swapaxes(chroma, 0, 1)[np.newaxis, :, :] # (1, 77, 12) TODO: figure out chunking
    chroma[chroma > 0] = 1 # normalise
    chroma = torch.from_numpy(chroma).type(torch.FloatTensor)
    generated = model.generate_with_chroma([args.caption], [chroma], sr)
    
    print(f"Generated {generated.shape}")