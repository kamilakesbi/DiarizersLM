from preprocessing.orchestrator import OrchestratorPipeline
import torch 
from datasets import Dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = Dataset.from_file("/data/fisher/generator/default-f61137895945b655/0.0.0/generator-train-00013-of-00059.arrow")

samples = dataset[:4]['audio']

orchestrator = OrchestratorPipeline.from_pretrained(
        asr_model = "openai/whisper-large-v3",
        diarizer_model = "pyannote/speaker-diarization-3.1", 
        device = device, 
    )

orchestrator(samples)

print('ok')
