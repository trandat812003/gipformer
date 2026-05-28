#!/usr/bin/env python3
'''
python -m src.trainning.compute_fbank
'''

import os
import logging
from pathlib import Path
import math
import pandas as pd
import sentencepiece as spm
import torch

from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
)

from icefall.utils import get_executor
from src.utils.nomalize_text import pre_process

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


INPUT_CSV = "./dataset/data.segments.csv"
AUDIO_DIR = "/media/trandat/Data"
OUTPUT_DIR = Path("./data/")
MANIFESTS_DIR = OUTPUT_DIR / "manifests"
FBANK_DIR = OUTPUT_DIR / "fbank"

MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
FBANK_DIR.mkdir(parents=True, exist_ok=True)

BPE_MODEL = "/media/trandat/Data/model/gipformer/bpe.model"
sp = spm.SentencePieceProcessor()
sp.load(BPE_MODEL)

csv_name = Path(INPUT_CSV).stem

def create_cutset(df):

    recordings = []
    supervisions = []

    for idx, row in df.iterrows():

        wav_path = Path(AUDIO_DIR) / str(row["file_path"])

        recording = Recording.from_file(wav_path)

        supervision = SupervisionSegment(
            id=str(idx),
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            text=str(pre_process(row["text"])),
        )

        recordings.append(recording)
        supervisions.append(supervision)

    return CutSet.from_manifests(
        recordings=RecordingSet.from_recordings(recordings),
        supervisions=SupervisionSet.from_segments(supervisions),
    )


def compute_fbank(
    df,
    name,
    perturb_speed=False,
):
    cut_set = create_cutset(df)

    if perturb_speed:
        cut_set = (
            cut_set
            + cut_set.perturb_speed(0.9)
            + cut_set.perturb_speed(1.1)
        )

    cut_set = cut_set.resample(16000)

    extractor = Fbank(FbankConfig(num_mel_bins=80))

    with get_executor() as ex:
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=FBANK_DIR / name,
            storage_type=LilcomChunkyWriter,
            num_jobs=15 if ex is None else 80,
            executor=ex,
        )

    cut_set.to_file(MANIFESTS_DIR / f"{name}.jsonl.gz")

    logging.info(f"Done: {name}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s "
            "[%(filename)s:%(lineno)d] %(message)s"
        ),
    )

    df = pd.read_csv(INPUT_CSV)

    n_train = 581
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    compute_fbank(train_df, "train", perturb_speed=True)
    compute_fbank(test_df, "test", perturb_speed=False)
