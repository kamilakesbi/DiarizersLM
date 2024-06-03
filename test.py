import torch
from asr_diarize import ASRDiarizationPipeline
from datasets import load_dataset

device = "cuda:2" if torch.cuda.is_available() else "cpu"
pipeline = ASRDiarizationPipeline.from_pretrained("openai/whisper-tiny", device=device)

# load dataset of concatenated LibriSpeech samples
dataset = load_dataset("diarizers-community/ami",'ihm', split="train", streaming=True)
# get first sample
sample = next(iter(dataset))

sample['audio']['array'] = sample['audio']['array'][:3*60*16000]

out_text, out_labels = pipeline(sample["audio"])

print(out_text, out_labels)