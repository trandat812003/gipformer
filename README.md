# requirements
```bash
pip install git+https://github.com/lhotse-speech/lhotse

cd /tmp
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=/tmp/icefall:$PYTHONPATH

uv pip install torch==1.13.0+cu116 torchaudio==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

uv pip install k2==1.24.3.dev20230725+cuda11.6.torch1.13.0 -f https://k2-fsa.github.io/k2/cuda.html

uv pip install git+https://github.com/lhotse-speech/lhotse
```

# preprocess
```bash
cd dataset/
awk -F',' 'BEGIN{OFS=","} NR==1{print; next} {$1="bidv/audio/"$1; print}' data.csv > tmp.csv && mv tmp.csv data.csv
awk -F',' 'BEGIN{OFS=","} NR==1{print; next} {$1=$1".wav"; print}' data.csv > tmp.csv && mv tmp.csv data.csv
```

# finetune
```bash
export PYTHONPATH=/home/trandat/Documents/gipformer/icefall:/home/trandat/Documents/gipformer/icefall/egs/librispeech/ASR:/home/trandat/Documents/gipformer/icefall/egs/librispeech/ASR/zipformer:$PYTHONPATH
```

```bash
INPUT_CSV=/home/trandat/Documents/gipformer/dataset/data.csv \
AUDIO_DIR=/media/trandat/Data \
python dataset/cut_audio.py
```

```bash
python dataset/fillter_tokens.py
```

```bash
python dataset/split_data.py
```

```bash
INPUT_CSV=/home/trandat/Documents/gipformer/dataset/data.gipformer.train.csv \
BPE_MODEL=/media/trandat/Data/model/gipformer/bpe.model \
AUDIO_DIR=/media/trandat/Data \
python -m src.trainning.compute_fbank
```

```bash
INPUT_CSV=/home/trandat/Documents/gipformer/dataset/data.gipformer.test.csv \
BPE_MODEL=/media/trandat/Data/model/gipformer/bpe.model \
AUDIO_DIR=/media/trandat/Data \
python -m src.trainning.compute_fbank
```

```bash
python -m src.trainning.gipformer \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --use-ctc 0 \
  --do-finetune 1 \
  --finetune-ckpt /media/trandat/Data/model/gipformer/epoch-35-avg-6.pt \
  --bpe-model /media/trandat/Data/model/gipformer/bpe.model \
  --manifest-dir /home/trandat/Documents/gipformer/data/manifests \
  --base-lr 0.0045 \
  --use-mux 0 \
  --exp-dir zipformer/exp_finetune \
  --max-duration 40
```