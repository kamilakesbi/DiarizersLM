import torch
from asr_diarize import ASRDiarizationPipeline
from datasets import load_dataset
import torch

# load dataset of concatenated LibriSpeech samples
dataset = load_dataset("diarizers-community/ami",'ihm', split="train", streaming=True)
# get first sample
sample = next(iter(dataset))

sample['audio']['array'] = sample['audio']['array'][:90*16000]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa" if is_torch_sdpa_available() else "eager"

pipeline = ASRDiarizationPipeline.from_pretrained(
    asr_model = "distil-whisper/distil-large-v3",
    diarizer_model = "pyannote/speaker-diarization-3.1", 
    llm_model = "meta-llama/Meta-Llama-3-8B",
    device=device,
)

hyp_text, hyp_labels = pipeline.orchestrate(sample['audio'])

print('orchestrate : done')

prompts = pipeline.generate_prompts(hyp_text, hyp_labels)

print('prompts generated')

completions = pipeline.generate_completions(prompts)

final_output = pipeline.post_process(completions, hyp_text, hyp_labels)

print(prompts)

print(final_output)