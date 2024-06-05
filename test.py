import torch
from asr_diarize import ASRDiarizationPipeline
from datasets import load_dataset
import torch
from diarizationlm import utils


# load dataset of concatenated LibriSpeech samples
dataset = load_dataset("diarizers-community/ami",'ihm', split="train", streaming=True)
# get first sample
sample = next(iter(dataset))

sample['audio']['array'] = sample['audio']['array'][:90*16000]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipeline = ASRDiarizationPipeline.from_pretrained("openai/whisper-tiny", device=device)

hyp_text, hyp_labels = pipeline.orchestrate(sample['audio'])

print('orchestrate : done')

prompts = pipeline.generate_prompts(hyp_text, hyp_labels)

print('prompts generated')

completions = pipeline.generate_completions(prompts)

final_output = pipeline.post_process(completions, hyp_text, hyp_labels)

print(prompts)

print(final_output)