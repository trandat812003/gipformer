'''
python dataset/cut_audio.py
'''

#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


INPUT_CSV = "/home/trandat/Documents/gipformer/dataset/data.csv"
AUDIO_DIR = "/media/trandat/Data"
OUTPUT_DIR = "/media/trandat/Data/bidv"
OUTPUT_CSV = "/home/trandat/Documents/gipformer/dataset/data.segments.csv"

output_audio_dir = Path(OUTPUT_DIR) / "audio_segments"
output_audio_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT_CSV)

new_rows = []


for idx, row in tqdm(df.iterrows(), total=len(df)):

    input_rel_path = str(row["file_path"])

    input_audio = Path(AUDIO_DIR) / input_rel_path

    start_time = float(row["startTime"])
    end_time = float(row["endTime"])
    channel = int(row["channel"])

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

        # chọn channel
        "-map_channel",
        f"0.0.{channel}",

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

new_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved CSV: {OUTPUT_CSV}")
print(f"Saved audio dir: {output_audio_dir}")