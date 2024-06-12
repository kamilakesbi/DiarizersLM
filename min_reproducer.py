from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor
import torch 
from torchaudio import functional as F
import numpy as np 

asr_model = "distil-whisper/distil-large-v3"
diarizer_model = "pyannote/speaker-diarization-3.1"

asr_processor = WhisperProcessor.from_pretrained(asr_model, token=True)
feature_extractor = WhisperFeatureExtractor.from_pretrained(asr_model, token=True)

raw_dataset = Dataset.from_file("/data/fisher/generator/default-f61137895945b655/0.0.0/generator-train-00013-of-00059.arrow").select(range(5))

sample = raw_dataset[0]['audio']

inputs = sample.pop("array", None)
in_sampling_rate = sample.pop("sampling_rate", None)

if in_sampling_rate != 16000:
    inputs = F.resample(torch.from_numpy(np.array(inputs)), in_sampling_rate, 16000).numpy()

# Whisper inputs: 
whisper_inputs = feature_extractor(inputs, sampling_rate=16000, truncation=False)

result = whisper_inputs.get('input_features')[0]

print(result)