from audiocraft.metrics import ChromaCosineSimilarityMetric
from audiocraft.models import MusicGen
from audiocraft.data.audio_utils import convert_audio, load_melody
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
    
    return midi_paths, generate(midi_paths, sr)

def process_melody(melody, device, sr, duration):
    target_sr = 32000
    target_ac = 1
    processed_melody = torch.from_numpy(melody).to(device).float().t()
    if processed_melody.dim() == 1:
        processed_melody = processed_melody[None]
    processed_melody = processed_melody[..., :int(sr * duration)]
    processed_melody = convert_audio(processed_melody, sr, target_sr, target_ac)
    return processed_melody

def evaluate(melody, description, model, duration, sr=32000):
    processed_melody = process_melody(melody, model.device, sr, duration)
    generated = model.generate_with_chroma([description], [processed_melody], sr)
    metric = ChromaCosineSimilarityMetric(sample_rate=sr, n_chroma=12, radix2_exp=12, argmax=False)
    generated_tensor = torch.tensor(generated[None, :])
    sample_rates = torch.tensor(np.array([sr])[None, :])
    sizes = torch.tensor(np.array([melody.shape[0]])[None, :]) # assuming size is n_frames here
    metric.update(generated_tensor, processed_melody, sizes, sample_rates)
    return metric.compute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to evaluation dataset")
    parser.add_argument("--model", type=str, default="facebook/musicgen-melody", help="path to evaluation dataset")
    parser.add_argument("--duration", type=int, default=10, help="duration of generated audio")
    args = parser.parse_args()

    # model prep
    model = MusicGen.get_pretrained(args.model)
    model.set_generation_params(duration=args.duration)
    
    # benchmark
    midi_paths, data_generator = extract_data(args.data_path, 32000)
    scores = []
    for melody, description in tqdm(data_generator, total=len(midi_paths)):
        scores.append(evaluate(melody, description, model, args.duration))
    pd.DataFrame({"path": midi_paths, "scores": scores}).to_csv(f"{os.path.dirname(args.data_path)}/results.csv")
    print(f"Score: {np.mean(scores)}±{np.std(scores)}")