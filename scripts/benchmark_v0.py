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

def walk_filter(input_dir, file_extension=None):
    files = []
    for r, _, fs in os.walk(input_dir, followlinks=True):
        if file_extension:
            files.extend([os.path.join(r, f) for f in fs if os.path.splitext(f)[-1] == file_extension])
        else:
            files.extend([os.path.join(r, f) for f in fs])
    return files

def extract_data(input_dir, captions_path, sr, mode="midi"):
    if mode == "midi":
        file_paths = walk_filter(input_dir, ".mid")
    else:
        file_paths = walk_filter(input_dir, ".mp3") + walk_filter(input_dir, ".wav")
    with open(captions_path) as captions_file:
        captions = captions_file.readlines()

    def generate(file_paths, captions, sr):
        for file_path in file_paths:
            for caption in captions:
                melody, _ = load_melody(file_path, sr)
                yield melody, file_path, caption
    
    return len(file_paths) * len(captions), generate(file_paths, captions, sr)

def process_melody(melody, device, sr, duration):
    target_sr = 32000
    target_ac = 1
    processed_melody = torch.from_numpy(melody).to(device).float().t()
    if processed_melody.dim() == 1:
        processed_melody = processed_melody[None]
    processed_melody = processed_melody[..., :int(sr * duration)]
    processed_melody = convert_audio(processed_melody, sr, target_sr, target_ac)
    return processed_melody

def evaluate(melody, caption, model, duration, sr=32000):
    processed_melody = process_melody(melody, model.device, sr, duration)
    generated = model.generate_with_chroma([caption], [processed_melody], sr)
    metric = ChromaCosineSimilarityMetric(sample_rate=sr, n_chroma=12, radix2_exp=12, argmax=False)
    sample_rates = torch.tensor(np.array([sr])[None, :])
    sizes = torch.tensor(np.array([melody.shape[0]])[None, :]) # assuming size is n_frames here
    metric.update(generated, processed_melody[None, :], sizes, sample_rates)
    return metric.compute(), generated

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to audio/midi dataset")
    parser.add_argument("--captions_path", type=str, required=True, help="path to captions file")
    parser.add_argument("--output_path", type=str, required=True, help="path to directory to save outputs to")
    parser.add_argument("--model", type=str, default="facebook/musicgen-melody", help="path to evaluation dataset")
    parser.add_argument("--duration", type=int, default=10, help="duration of generated audio")
    parser.add_argument("--mode", type=str, default="midi", choices=["midi", "audio"], help="duration of generated audio")
    args = parser.parse_args()

    # model prep
    model = MusicGen.get_pretrained(args.model)
    model.set_generation_params(duration=args.duration)
    
    # benchmark
    pathlib.Path(args.output_path).mkdir(exist_ok=True, parents=True)
    n_total, data_generator = extract_data(args.data_path, args.captions_path, 32000, args.mode)
    rows = []
    for melody, path, caption in tqdm(data_generator, total=n_total):
        # evaluate
        score, generated = evaluate(melody, caption, model, args.duration)
        rows.append([path, caption, score])
        # save output
        basename = os.path.splitext(os.path.basename(path))[0]
        idx = len(rows)-1
        audio_write(f'{args.output_path}/{basename}_{idx}', generated.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

    df = pd.DataFrame(rows, columns=["path", "caption", "score"])
    df.to_csv(f"{os.path.dirname(args.data_path)}/results.csv")
    print(f'Score: {df["score"].mean()}±{df["score"].std()}')