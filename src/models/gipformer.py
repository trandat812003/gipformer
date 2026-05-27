#!/usr/bin/env python3
"""
export PYTHONPATH=/home/trandat/Documents/gipformer/icefall:/home/trandat/Documents/gipformer/icefall/egs/librispeech/ASR:/home/trandat/Documents/gipformer/icefall/egs/librispeech/ASR/zipformer:$PYTHONPATH

AUDIO_FILE=/home/trandat/Documents/gipformer/audio_114.wav \
python src/models/gipformer.py

"""

import math
import os
import time
import warnings
from pathlib import Path

import k2
import kaldifeat
import sentencepiece as spm
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SAMPLE_RATE = 16000

MODEL_DIR = Path("/media/trandat/Data/model/gipformer")

model_paths = {
    "checkpoint": str(MODEL_DIR / "epoch-35-avg-6.pt"),
    "bpe_model": str(MODEL_DIR / "bpe.model"),
    "tokens": str(MODEL_DIR / "tokens.txt"),
}


def load_model():
    from egs.librispeech.ASR.zipformer.train import get_model, get_params

    params = get_params()
    params.update({
        "context_size": 2,
        "num_encoder_layers": '2,2,3,4,3,2',
        "downsampling_factor": '1,2,4,8,4,2', 
        'feedforward_dim': '512,768,1024,1536,1024,768', 
        'num_heads': '4,4,4,8,4,4', 
        'encoder_dim': '192,256,384,512,384,256', 
        'query_head_dim': '32', 
        'value_head_dim': '12', 
        'pos_head_dim': '4', 
        'pos_dim': 48, 
        'encoder_unmasked_dim': '192,192,256,256,256,192', 
        'cnn_module_kernel': '31,31,15,15,15,31', 
        'decoder_dim': 512, 
        'joiner_dim': 512, 
        'attention_decoder_dim': 512, 
        'attention_decoder_num_layers': 6, 
        'attention_decoder_attention_dim': 512, 
        'attention_decoder_num_heads': 8, 
        'attention_decoder_feedforward_dim': 2048, 
        'causal': False, 
        'chunk_size': '16,32,64,-1', 
        'left_context_frames': '64,128,256,-1', 
        'use_transducer': True, 
        'use_ctc': False, 
        'use_attention_decoder': False, 
        'use_cr_ctc': False
    })


    # Download model files
    model_paths = {
        "checkpoint": str(MODEL_DIR / "epoch-35-avg-6.pt"),
        "bpe_model": str(MODEL_DIR / "bpe.model"),
        "tokens": str(MODEL_DIR / "tokens.txt"),
    }

    # Token table (for blank_id and vocab_size)
    token_table = k2.SymbolTable.from_file(model_paths["tokens"])
    params.blank_id = token_table["<blk>"]
    params.unk_id = token_table["<unk>"]

    # Count tokens excluding disambiguation symbols
    num_tokens = 0
    for s in token_table.symbols:
        if not s.startswith("#"):
            num_tokens += 1
    # Exclude token ID 0 (blank) from count
    if token_table["<blk>"] == 0:
        num_tokens -= 1
    params.vocab_size = num_tokens + 1

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build and load model
    model = get_model(params)

    checkpoint = torch.load(
        model_paths["checkpoint"], map_location="cpu", weights_only=False
    )
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Sentencepiece for text decoding
    sp = spm.SentencePieceProcessor()
    sp.load(model_paths["bpe_model"])

    # Feature extractor
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    fbank = kaldifeat.Fbank(opts)

    return model, sp, fbank, device


def decode(model, encoder_out, encoder_out_lens, beam_size=4, decoding_method="modified_beam_search"):
    from egs.librispeech.ASR.zipformer.beam_search import greedy_search_batch, modified_beam_search

    if decoding_method == "greedy_search":
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
    else:
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=beam_size,
        )

    return hyp_tokens

# ── Main ─────────────────────────────────────────────────────────────────────


def main():

    import torch
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence

    audio_file = os.environ["AUDIO_FILE"]

    model, sp, fbank, device = load_model()

    # Transcribe each audio file
    with torch.no_grad():
        start = time.time()

        # Load audio, resample if needed
        wave, sr = torchaudio.load(audio_file)
        if sr != SAMPLE_RATE:
            wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
        wave = wave[0].contiguous().to(device)
        audio_duration = wave.shape[0] / SAMPLE_RATE

        # Extract features
        features = fbank([wave])
        feature_lengths = torch.tensor([features[0].size(0)], device=device)
        features = pad_sequence(
            features, batch_first=True, padding_value=math.log(1e-10)
        )

        # Encode
        encoder_out, encoder_out_lens = model.forward_encoder(
            features, feature_lengths
        )

        # Decode
        hyp_tokens = decode(model, encoder_out, encoder_out_lens)

        # Convert tokens to text
        text = sp.decode(hyp_tokens[0])

        elapsed = time.time() - start
        rtf = elapsed / audio_duration if audio_duration > 0 else 0

        print(f"\n  File: {audio_file}")
        print(f"  Text: {text}")
        print(f"  Time: {elapsed:.2f}s | Audio: {audio_duration:.2f}s | RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()