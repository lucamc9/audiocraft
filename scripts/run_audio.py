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

def process_melody(melody, device, sr, duration):
    target_sr = 32000
    target_ac = 1
    processed_melody = torch.from_numpy(melody).to(device).float().t()
    if processed_melody.dim() == 1:
        processed_melody = processed_melody[None]
    processed_melody = processed_melody[..., :int(sr * duration)]
    processed_melody = convert_audio(processed_melody, sr, target_sr, target_ac)
    return processed_melody

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True, help="path to audio file")
    parser.add_argument("--caption", type=str, required=True, help="caption")
    args = parser.parse_args()
    
    sr = 32000
    duration = 10

    # model prep
    model = MusicGen.get_pretrained("facebook/musicgen-melody")
    model.set_generation_params(duration=duration)

    # inference
    melody, _ = load_melody(args.audio_path, sr)
    processed_melody = process_melody(melody, model.device, sr, duration)
    generated = model.generate_with_chroma([args.caption], [processed_melody], sr)
    
    print(f"Generated {generated.shape}")