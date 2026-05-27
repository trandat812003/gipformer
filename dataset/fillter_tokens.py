"""
python dataset/fillter_tokens.py
"""

import re
import pandas as pd
from tqdm import tqdm
from functools import lru_cache

TOKEN_FILE = "/media/trandat/Data/model/gipformer/tokens.txt"
INPUT_CSV = "/home/trandat/Documents/gipformer/dataset/data.segments.csv"
OUTPUT_CSV = INPUT_CSV.replace(".segments.csv", ".gipformer.csv")
TEXT_COLUMN = "text"

# =========================
# LOAD TOKENS
# =========================
with open(TOKEN_FILE, "r", encoding="utf-8") as f:
    tokens = set(line.strip() for line in f if line.strip())

# để tối ưu
tokens = set()
with open(TOKEN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        # lấy token đầu tiên
        tok = line.split()[0]

        # bỏ ▁
        tok = tok.replace("▁", "")

        tok = tok.strip()

        if tok:
            tokens.add(tok.lower())

# breakpoint()

@lru_cache(maxsize=None)
def dp(word):
    word = word.lower().strip()

    # tránh recursion vô hạn
    if word == "":
        return False

    if word in tokens:
        return True

    n = len(word)

    for i in range(1, n):
        if dp(word[:i]) and dp(word[i:]):
            return True

    return False


df = pd.read_csv(INPUT_CSV)

new_texts = []

for text in tqdm(df[TEXT_COLUMN]):
    if pd.isna(text):
        new_texts.append("")
        continue

    text = str(text).strip()

    if text == "":
        new_texts.append("")
        continue

    text = re.sub(r"[_\W]+", " ", str(text.lower()), flags=re.UNICODE)
    words = text.split()

    kept_words = []

    for word in words:
        word = word.strip()

        if not word:
            continue

        if dp(word):
            kept_words.append(word)
        else:
            print(word)

    new_texts.append(" ".join(kept_words))

df["gipformer"] = new_texts
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved to: {OUTPUT_CSV}")