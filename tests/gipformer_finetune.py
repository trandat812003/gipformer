import torch
import torchaudio

from src.models.gipformer import load_model, decode
from src.utils.nomalize_text import normalize_text

# ===== load model =====
model, sp, fbank, device = load_model()

print(model)

ckpt = torch.load(
    "/home/trandat/Downloads/best-valid-loss.pt",
    map_location=device,
    weights_only=False,
)

model.load_state_dict(ckpt["model"], strict=False)

model.eval()
model.to(device)

# ===== audio =====
waveform, sr = torchaudio.load(
    "/media/trandat/Data/bidv/audio_segments/1_16.wav"
)

# mono
waveform = waveform.mean(dim=0)

if sr != 16000:
    waveform = torchaudio.functional.resample(
        waveform, sr, 16000
    )

waveform = waveform.to(device)

# ===== feature extraction =====
features = fbank([waveform])

feature_lengths = torch.tensor(
    [features[0].size(0)],
    device=device,
)

# pad batch (quan trọng giống code test của bạn)
import math
from torch.nn.utils.rnn import pad_sequence

features = pad_sequence(
    features,
    batch_first=True,
    padding_value=math.log(1e-10),
)

# ===== inference (USE OFFICIAL PIPELINE) =====
with torch.no_grad():

    encoder_out, encoder_out_lens = model.forward_encoder(
        features,
        feature_lengths,
    )

    hyp_tokens = decode(
        model,
        encoder_out,
        encoder_out_lens,
    )

    text = sp.decode(hyp_tokens[0]).strip()

print("PRED:", normalize_text(text))