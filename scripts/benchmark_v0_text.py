from audiocraft.metrics import CLAPTextConsistencyMetric
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

def evaluate(caption, model, sr=32000):
    generated = model.generate([caption])
    sample_rates = torch.tensor(np.array([sr])[None, :])
    sizes = torch.tensor(np.array([generated.shape[0]])[None, :]) # assuming size is n_frames here
    clap_metric = CLAPTextConsistencyMetric()
    clap_metric.update(generated, [caption], sizes, sample_rates)
    return clap_metric.compute(), generated[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_path", type=str, required=True, help="path to captions file")
    parser.add_argument("--output_path", type=str, required=True, help="path to directory to save outputs to")
    parser.add_argument("--model", type=str, default="facebook/musicgen-melody", help="path to evaluation dataset")
    parser.add_argument("--duration", type=int, default=10, help="duration of generated audio")
    parser.add_argument("--instrument", type=int, default=4, choices=range(1,131), metavar="[1-130]", help="instrument program [1-130] to synthesize with")
    args = parser.parse_args()

    # model prep
    model = MusicGen.get_pretrained(args.model)
    model.set_generation_params(duration=args.duration)
    
    # benchmark
    pathlib.Path(args.output_path).mkdir(exist_ok=True, parents=True)
    with open(args.captions_path) as captions_file:
        captions = captions_file.readlines()
    
    for caption in tqdm(captions, total=len(captions)):
        # evaluate
        clap_score, generated = evaluate(caption, model)
        rows.append([caption, clap_score])
        # save output
        basename = "_".join(caption.split())
        idx = len(rows)-1
        audio_write(f'{args.output_path}/{basename}', generated.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

    df = pd.DataFrame(rows, columns=["caption", "score"])
    df.to_csv(f"{args.output_path}/results.csv")
    print(f'Score: {df["score"].mean()}±{df["score"].std()}')