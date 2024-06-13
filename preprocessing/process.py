from typing import Any
from datasets import Dataset
import re 
from diarizationlm import utils
from orchestrator import OrchestratorPipeline
import torch 
from multiprocess import set_start_method
from transformers import WhisperTokenizer


class Preprocess: 

    def __init__(
        self,
        orchestrator,
        normalizer, 
    ) -> None:
        
        set_start_method('spawn')

        self.prompt_options = utils.PromptOptions()
        self.speaker_prefix = self.prompt_options.speaker_prefix
        self.speaker_suffix = self.prompt_options.speaker_suffix

        self.orchestrator = orchestrator
        self.normalizer = normalizer

    def normalize_text(self, text): 

        text = self.normalizer.normalize(text)

        return text
    
    def __call__(self, transcripts_column, speakers_column, audio_column, rank):

        device = "cuda:" + str(rank % torch.cuda.device_count())
        new_batch = {"ref_diarized_text": [], 'ref_text': [], 'ref_labels': [], 'hyp_text': [], 'hyp_labels': [], 'hyp_diarized_text': []} 

        self.orchestrator.to_device(device)

        hyp_text_list, hyp_labels_list = self.orchestrator(audio_column) 
        
        hyp_diarized_text_list = []
        for i in range(len(hyp_text_list)): 
            hyp_diarized_text_list.append(utils.create_diarized_text(hyp_text_list[i].split(' '), hyp_labels_list[i].split(' ')))

        ref_diarized_text = ''
        for i, transcriptions in enumerate(transcripts_column):
            
            # Map speakers to integer values as required by diarizationlm:
            speaker_to_int = {speaker: str(idx + 1) for idx, speaker in enumerate(sorted(set(speakers_column[i])))}
            speakers = [speaker_to_int[speaker] for speaker in speakers_column[i]]

            for index, transcript in enumerate(transcriptions):
                ref_diarized_text += self.speaker_prefix + speakers[index] + self.speaker_suffix + ' '
                ref_diarized_text += self.normalize_text(transcript)
                ref_diarized_text += ' '

            ref_text, ref_labels = utils.extract_text_and_spk(ref_diarized_text, po=self.prompt_options)

            new_batch['ref_diarized_text'].append(ref_diarized_text)
            new_batch['ref_text'].append(ref_text)
            new_batch['ref_labels'].append(ref_labels)

            new_batch['hyp_diarized_text'].append(hyp_diarized_text_list[i])
            new_batch['hyp_text'].append(hyp_text_list[i])
            new_batch['hyp_labels'].append(hyp_labels_list[i])

        return new_batch


if __name__ == '__main__': 

    dataset = Dataset.from_file("/data/fisher/generator/default-f61137895945b655/0.0.0/generator-train-00013-of-00059.arrow")
    

    orchestrator = OrchestratorPipeline.from_pretrained(
        asr_model = "distil-whisper/distil-large-v3",
        diarizer_model = "pyannote/speaker-diarization-3.1", 
    )

    normalizer = WhisperTokenizer.from_pretrained("distil-whisper/distil-large-v3")

    # Load the preprocessor: 
    preprocessor = Preprocess(orchestrator, normalizer)


    dataset = dataset.map(
        preprocessor, 
        input_columns=['transcripts', 'speakers', 'audio'], 
        batched=True, 
        batch_size=4,
        remove_columns=['transcripts', 'speakers'],  
        with_rank=True,
        num_proc=20,
    )

    print(dataset)








