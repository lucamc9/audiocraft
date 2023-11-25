from audiocraft.metrics import ChromaCosineSimilarityMetric
from audiocraft.models import MusicGen
from audiocraft.data.audio_utils import load_melody
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import librosa
import torch
import os

def walk_filter(input_dir, file_extension=None):
    files = []
    for r, _, fs in os.walk(input_dir, followlinks=True):
        if file_extension:
            files.extend([os.path.join(r, f) for f in fs if os.path.splitext(f)[-1] == file_extension])
        else:
            files.extend([os.path.join(r, f) for f in fs])
    return files

def extract_data(input_dir, sr):
    midi_paths = walk_filter(input_dir, ".mid")

    def generate(midi_paths, sr):
        for midi_path in midi_paths:
            melody, _ = load_melody(midi_path, sr)
            description_file = open(f"{os.path.splitext(midi_path)[0]}.txt")
            description = description_file.read()
            description_file.close()
            yield melody, description
    
    return generate(midi_paths, sr), len(midi_paths)

def evaluate(melody, description, model, sr):
    melody_tensor = torch.tensor(melody)
    generated = model.generate_with_chroma([description], melody[None].expand(3, -1, -1), sr)
    # melody = melody[:generated.shape[0]] # make sure they're the same dimensions
    metric = ChromaCosineSimilarityMetric(sample_rate=sr, n_chroma=12, radix2_exp=12, argmax=False)
    generated_tensor = torch.tensor(generated[None, :])
    sample_rates = torch.tensor(np.array([sr])[None, :])
    sizes = torch.tensor(np.array([melody.shape[0]])[None, :]) # assuming size is n_frames here
    metric.update(generated_tensor, melody_tensor[None, :], sizes, sample_rates)
    return metric.compute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to evaluation dataset")
    parser.add_argument("--model", type=str, default="facebook/musicgen-melody", help="path to evaluation dataset")
    parser.add_argument("--duration", type=int, default=10, help="duration of generated audio")
    parser.add_argument("--sr", type=int, default=32000, help="default sample rate")
    args = parser.parse_args()

    # model prep
    model = MusicGen.get_pretrained(args.model)
    model.set_generation_params(duration=args.duration)
    
    # benchmark
    data_generator, n_samples = extract_data(args.data_path, args.sr)
    scores = []
    for melody, description in tqdm(data_generator, total=n_samples):
        scores.append(evaluate(melody, description, model, args.sr))
    pd.DataFrame({"path": midi_paths, "scores": scores}).to_csv(f"{os.path.dirname(args.data_path)}/results.csv")
    print(f"Score: {np.mean(scores)}±{np.std(scores)}")