# DiarizersLM

DiarizersLM is an experimental project inspired by Google's DiarizationLM work.
It reimplements the diarization-correction pipeline with Whisper for automatic
speech recognition (ASR), pyannote.audio for speaker diarization, and a
causal-language model for speaker-assignment correction.

## How it works

The pipeline first uses pyannote.audio to identify speaker turns and their time
spans. Whisper transcribes the same audio and returns timestamped text chunks.
The orchestration stage aligns each Whisper chunk with the diarization segment
having the greatest temporal overlap, producing a transcript paired with an
initial speaker label for every word. If a chunk has no overlap, it is assigned
to the nearest diarized turn.

This initial transcript can contain speaker mistakes because ASR and
diarization boundaries do not always line up. DiarizersLM converts it into
prompts and sends them to an LLM, which corrects misplaced words and their
speaker assignments. Finally, the corrected speaker labels are transferred
back onto the original ASR transcript to produce the diarized output.

## Status and requirements

This is an experimental, GPU-oriented project. It requires Python, PyTorch
compatible with the installed CUDA runtime, FFmpeg (for file inference), and
the packages in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

The default diarization and Llama models are gated on Hugging Face. Authenticate
with an account that has accepted each model's terms before running the
preprocessing or inference commands.

## Dataset construction

Obtain and unpack the licensed Fisher corpus yourself, then construct a local
Hugging Face dataset. Paths below are examples; choose writable locations.

```bash
python preprocessing/construct.py \
  --local_fisher_dir /path/to/fisher/data \
  --preprocess_cache_dir /path/to/cache \
  --hub_folder your-account/fisher
```

Generate ASR hypotheses and diarization labels:

```bash
accelerate launch --num_processes 4 preprocessing/run.py \
  --asr_name_or_path distil-whisper/distil-large-v3 \
  --diarizer_name_or_path pyannote/speaker-diarization-3.1 \
  --dataset_name your-account/fisher \
  --dataset_split_name train \
  --per_device_batch_size 4 \
  --dataloader_num_workers 4 \
  --num_proc 12 \
  --dtype bfloat16 \
  --push_to_hub \
  --output_hub_repository your-account/processed_fisher
```

Add oracle and degraded labels to one processed split:

```bash
python preprocessing/run_oracle_deg.py \
  --dataset_name your-account/processed_fisher \
  --dataset_split_name train \
  --num_proc 12 \
  --push_to_hub \
  --output_hub_repository your-account/processed_fisher
```

## Training

`training/train.py` is the standard TRL/PEFT training entry point. Update its
dataset and model identifiers for your Hub account and accepted models.

The Unsloth workflow reads file paths from `train_unsloth/config.py`. Create
the JSON inputs with `train_unsloth/prepare_for_unsloth.py`, set the paths in
that configuration, then run:

```bash
python train_unsloth/fine_tune.py
```

## Inference

Instantiate `DiarizersLmPipeline` with model IDs you can access, and pass an
audio path, audio bytes, a one-dimensional NumPy array, or a datasets-style
audio dictionary. `test.py` contains a minimal Hub-dataset example.

```python
from inference.pipeline import DiarizersLmPipeline

pipeline = DiarizersLmPipeline.from_pretrained(device="cuda:0")
result = pipeline("/path/to/audio.wav")
```

Use a CPU device only for small experiments; the diarization and LLM models are
resource intensive.
