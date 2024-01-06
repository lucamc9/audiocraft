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

def get_chroma(midi):
    # 1. extract chroma with 79 timesteps to match MusicGen's chroma
    chroma = midi.get_chroma(fs=79/midi.get_end_time()) # desired_n * 1/fs = end_time
    
    # 2. argmax + normalise
    norm_chroma = torch.Tensor(chroma.copy().T)
    idx = norm_chroma.argmax(-1, keepdim=True)
    norm_chroma[:] = 0
    norm_chroma.scatter_(dim=-1, index=idx, value=1)
    
    # 3. add and fill second dimension with C1
    norm_chroma = torch.unsqueeze(norm_chroma, 0)
    C1_array = np.zeros((1, 79, 12))
    C1_array[0, :, 0] = 1
    norm_chroma = torch.cat((norm_chroma, torch.Tensor(C1_array)))
    
    return norm_chroma

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

    # inference (first attempt)
    # melody = pretty_midi.PrettyMIDI(args.midi_path)
    # chroma = melody.get_chroma(fs=235/30) # (12, 77)
    # chroma = np.swapaxes(chroma, 0, 1) # (77, 12) TODO: figure out chunking
    # chroma[chroma > 0] = 1 # normalise
    
    # TODO: remove after quick test
    # chroma = np.load(args.midi_path)
    # chroma = torch.from_numpy(chroma).type(torch.FloatTensor)

    # inference (second attempt)
    midi = pretty_midi.PrettyMIDI(args.midi_path)
    chroma = get_chroma(midi)
    generated = model.generate_with_chroma([args.caption], [chroma], sr)
    audio_write(f'{os.path.dirname(args.midi_path)}/generated', generated[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    
    print(f"Generated {generated.shape}")