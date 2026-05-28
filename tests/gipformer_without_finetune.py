'''
export PYTHONPATH=/home/trandat/Documents/gipformer/icefall:/home/trandat/Documents/gipformer/icefall/egs/librispeech/ASR:/home/trandat/Documents/gipformer/icefall/egs/librispeech/ASR/zipformer:$PYTHONPATH

AUDIO_DIR=/media/trandat/Data \
INPUT_CSV=/home/trandat/Documents/gipformer/dataset/data.gipformer.test.csv \
python -m tests.gipformer_without_finetune
'''

import os
import math
from datetime import datetime
from tqdm import tqdm

import torch
import torchaudio
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from jiwer import wer

from src.models.gipformer import (
    load_model,
    SAMPLE_RATE,
    decode,
)
from src.utils.nomalize_text import normalize_text


def main():
    AUDIO_DIR = "/media/trandat/Data"
    input_csv = "/home/trandat/Documents/gipformer/dataset/data.segments.csv"
    output_csv = f"./outputs/{datetime.now().strftime('%Y%m%d_%H-%M')}/predictions.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(input_csv).fillna("")
    df = df.iloc[581:]

    model, sp, fbank, device = load_model()

    predictions = []
    references = []

    with torch.no_grad():
        for i, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Inferencing",
        ):
            audio_path = row["file_path"]
            reference = str(row["text"]).strip()

            try:
                # load audio
                wave, sr = torchaudio.load(AUDIO_DIR + "/" + audio_path)

                if sr != SAMPLE_RATE:
                    wave = torchaudio.functional.resample(
                        wave,
                        sr,
                        SAMPLE_RATE,
                    )

                wave = wave[0].contiguous().to(device)

                # extract feature
                features = fbank([wave])

                feature_lengths = torch.tensor(
                    [features[0].size(0)],
                    device=device,
                )

                features = pad_sequence(
                    features,
                    batch_first=True,
                    padding_value=math.log(1e-10),
                )

                # encoder
                encoder_out, encoder_out_lens = model.forward_encoder(
                    features,
                    feature_lengths,
                )

                # decode
                hyp_tokens = decode(
                    model,
                    encoder_out,
                    encoder_out_lens,
                )

                pred_text = sp.decode(hyp_tokens[0]).strip()

            except Exception as e:
                print(f"ERROR: {audio_path}")
                print(e)
                pred_text = ""

            predictions.append(normalize_text(pred_text))
            references.append(normalize_text(reference))

            # print(f"\n[{i}]")
            # print("REF :", reference)
            # print("PRED:", pred_text)

    # save result
    df["prediction"] = predictions

    df.to_csv(output_csv, index=False)

    # corpus WER
    total_wer = wer(references, predictions)

    print("\n==============================")
    print(f"WER: {total_wer * 100:.2f}%")
    print(f"Saved: {output_csv}")
    print("==============================")


if __name__ == "__main__":
    main()