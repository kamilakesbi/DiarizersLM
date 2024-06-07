from preprocessing.orchestrator import OrchestratorPipeline
import torch 

device = "cuda:0" if torch.cuda.is_available() else "cpu"


orchestrator = OrchestratorPipeline.from_pretrained(
        asr_model = "openai/whisper-large-v3",
        diarizer_model = "pyannote/speaker-diarization-3.1", 
        device = device, 
    )

print(orchestrator.asr_pipeline.device)
print(orchestrator.diarization_pipeline.device)
