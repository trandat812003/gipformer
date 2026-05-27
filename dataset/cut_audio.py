'''
INPUT_CSV=/home/trandat/Documents/gipformer/dataset/data.csv \
AUDIO_DIR=/media/trandat/Data \
python dataset/cut_audio.py
'''

#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


INPUT_CSV = os.environ["INPUT_CSV"]
AUDIO_DIR = os.environ.get("AUDIO_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/media/trandat/Data/bidv")


output_audio_dir = Path(OUTPUT_DIR) / "audio_segments"
output_audio_dir.mkdir(parents=True, exist_ok=True)


csv_name = Path(INPUT_CSV).stem
output_csv = INPUT_CSV.replace(".csv", ".segments.csv")


df = pd.read_csv(INPUT_CSV)

new_rows = []


for idx, row in tqdm(df.iterrows(), total=len(df)):

    input_rel_path = str(row["file_path"])

    input_audio = Path(AUDIO_DIR) / input_rel_path

    start_time = float(row["startTime"])
    end_time = float(row["endTime"])

    duration = end_time - start_time

    if duration <= 0:
        continue

    original_stem = Path(input_rel_path).stem

    output_name = f"{original_stem}_{idx}.wav"

    output_audio = output_audio_dir / output_name

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-ss",
        str(start_time),
        "-to",
        str(end_time),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_audio),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    new_row = row.to_dict()

    new_row["file_path"] = f"bidv/audio_segments/{output_name}"
    new_row["duration"] = duration

    new_rows.append(new_row)


new_df = pd.DataFrame(new_rows)

new_df.to_csv(output_csv, index=False)

print(f"Saved CSV: {output_csv}")
print(f"Saved audio dir: {output_audio_dir}")

print(f"Saved audio dir: {output_audio_dir}")