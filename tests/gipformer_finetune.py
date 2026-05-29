'''
python -m tests.gipformer_finetune
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
    INPUT_CSV = "/home/trandat/Documents/gipformer/dataset/data.segments.csv"
    CKPT_PATH = "/home/trandat/Downloads/best-valid-loss.pt"
    USE_CTC = True
    USE_TRANSDUCER = True
    INFER_BATCH_SIZE = 32

    run_id = datetime.now().strftime('%Y%m%d_%H-%M')
    output_dir = f"./outputs/{run_id}"
    output_csv = f"{output_dir}/predictions.csv"
    output_log = f"{output_dir}/eval.log"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(INPUT_CSV).fillna("")
    df = df.iloc[581:].reset_index(drop=True)

    model, sp, fbank, device = load_model(
        use_ctc=USE_CTC,
        use_transducer=USE_TRANSDUCER,
    )

    ckpt = torch.load(
        CKPT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    model.to(device)

    predictions = [""] * len(df)
    references = [normalize_text(str(t).strip()) for t in df["text"].tolist()]

    def decode_feature_batch(feature_list):
        feature_lengths = torch.tensor(
            [f.size(0) for f in feature_list],
            device=device,
        )

        features = pad_sequence(
            feature_list,
            batch_first=True,
            padding_value=math.log(1e-10),
        )

        encoder_out, encoder_out_lens = model.forward_encoder(
            features,
            feature_lengths,
        )

        hyp_tokens = decode(
            model,
            encoder_out,
            encoder_out_lens,
        )

        return [
            normalize_text(sp.decode(tokens).strip())
            for tokens in hyp_tokens
        ]

    batch_features = []
    batch_indices = []

    def flush_batch():
        nonlocal batch_features, batch_indices
        if not batch_features:
            return

        try:
            batch_preds = decode_feature_batch(batch_features)
            for idx, pred in zip(batch_indices, batch_preds):
                predictions[idx] = pred
        except Exception as e:
            print(f"BATCH ERROR ({len(batch_features)} items): {e}")
            # Fallback to single-item decode so one bad sample won't drop a whole batch.
            for idx, feat in zip(batch_indices, batch_features):
                try:
                    predictions[idx] = decode_feature_batch([feat])[0]
                except Exception as single_e:
                    print(f"ERROR (idx={idx}): {single_e}")
                    predictions[idx] = ""

        batch_features = []
        batch_indices = []

    with torch.no_grad():
        for i, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Inferencing",
        ):
            audio_path = row["file_path"]

            try:
                wave, sr = torchaudio.load(AUDIO_DIR + "/" + audio_path)

                if sr != SAMPLE_RATE:
                    wave = torchaudio.functional.resample(
                        wave,
                        sr,
                        SAMPLE_RATE,
                    )

                wave = wave.mean(dim=0).contiguous().to(device)
                feature = fbank([wave])[0]

                batch_features.append(feature)
                batch_indices.append(i)

                if len(batch_features) >= INFER_BATCH_SIZE:
                    flush_batch()

            except Exception as e:
                print(f"ERROR: {audio_path}")
                print(e)
                predictions[i] = ""

        flush_batch()

    df["prediction"] = predictions
    df.to_csv(output_csv, index=False)

    total_wer = wer(references, predictions)

    with open(output_log, "a", encoding="utf-8") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"ckpt: {CKPT_PATH}\n")
        f.write(f"use_ctc: {USE_CTC}\n")
        f.write(f"use_transducer: {USE_TRANSDUCER}\n")
        f.write(f"infer_batch_size: {INFER_BATCH_SIZE}\n")
        f.write(f"num_samples: {len(df)}\n")
        f.write(f"wer: {total_wer * 100:.2f}%\n")
        f.write(f"prediction_csv: {output_csv}\n")
        f.write("-" * 40 + "\n")

    print("\n==============================")
    print(f"WER: {total_wer * 100:.2f}%")
    print(f"Saved: {output_csv}")
    print(f"Log: {output_log}")
    print("==============================")


if __name__ == "__main__":
    main()