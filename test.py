import torch
from pipeline import DiarizersLmPipeline
from datasets import load_dataset
import torch
from scipy.io.wavfile import write

# load dataset of concatenated LibriSpeech samples
dataset = load_dataset("diarizers-community/ami",'ihm', split="train", streaming=True)
# get first sample
sample = next(iter(dataset))

sample['audio']['array'] = sample['audio']['array'][60*16000:3*60*16000]

audio = write( filename='example.wav', rate=16000, data=sample['audio']['array'])

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipeline = DiarizersLmPipeline.from_pretrained(
    asr_model = "openai/whisper-large-v3",
    diarizer_model = "pyannote/speaker-diarization-3.1", 
    llm_model = "meta-llama/Meta-Llama-3-8B",
    device=device, 
)

output = pipeline(sample['audio'])
