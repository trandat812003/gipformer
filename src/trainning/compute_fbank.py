#!/usr/bin/env python3
'''
export PYTHONPATH=/home/trandat/Documents/gipformer/icefall:/home/trandat/Documents/gipformer/icefall/egs/librispeech/ASR:/home/trandat/Documents/gipformer/icefall/egs/librispeech/ASR/zipformer:$PYTHONPATH

INPUT_CSV=/home/trandat/Documents/gipformer/dataset/data.gipformer.train.csv \
BPE_MODEL=/media/trandat/Data/model/gipformer/bpe.model \
AUDIO_DIR=/media/trandat/Data \
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

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


INPUT_CSV = os.environ["INPUT_CSV"]
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data/")
BPE_MODEL = os.environ.get("BPE_MODEL", "/media/trandat/Data/model/gipformer/bpe.model")
AUDIO_DIR = os.environ.get("AUDIO_DIR", "/media/trandat/Data")

os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
csv_name = Path(INPUT_CSV).stem

def create_cutset_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    recordings = []
    supervisions = []

    for idx, row in df.iterrows():
        wav_path = AUDIO_DIR + "/" +row["file_path"]

        recording = Recording.from_file(
            wav_path,
            # recording_id=str(idx),
        )

        supervision = SupervisionSegment(
            id=str(idx),
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            text=str(row["text"]),
            # channel=int(row["channel"]),
        )

        recordings.append(recording)
        supervisions.append(supervision)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    cut_set = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set,
    )

    return cut_set


def compute_fbank(
    csv_path,
    output_dir,
    bpe_model=None,
    perturb_speed=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    num_jobs = 15
    num_mel_bins = 80

    if bpe_model:
        logging.info(f"Loading {bpe_model}")

        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    logging.info("Creating CutSet from CSV")
    cut_set = create_cutset_from_csv(csv_path)

    if perturb_speed:
        logging.info("Applying speed perturb")

        cut_set = (
            cut_set
            + cut_set.perturb_speed(0.9)
            + cut_set.perturb_speed(1.1)
        )

    cut_set = cut_set.resample(16000)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=output_dir / "fbank" / csv_name,
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )

    manifest_path = manifests_dir / f"{csv_name}.jsonl.gz"
    cut_set.to_file(manifest_path)

    logging.info("Done")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s "
        "[%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank(
        csv_path=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        bpe_model=BPE_MODEL,
        perturb_speed=True,
    )