from datasets import load_dataset, IterableDatasetDict, Dataset
from diarizationlm import utils
import torch 
from multiprocess import set_start_method
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor
from pyannote.audio import Pipeline

from transformers.utils import is_torch_sdpa_available 
from accelerate import Accelerator
from torchaudio import functional as F
import numpy as np 
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union
from dataclasses import dataclass


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


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Any
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:

        batch = {}

        batch['timestamps_start'] = [f['timestamps_start'] for f in features]
        batch['timestamps_end'] = [f['timestamps_end'] for f in features]
        batch['speakers'] = [f['speakers'] for f in features]
        batch['transcripts'] = [f['transcripts'] for f in features] 
        whisper_inputs = {'inputs_features': [f['whisper_inputs']['input_features'] for f in features]}

        # reformat list to dict and set to pytorch format
        batch['whisper_inputs'] = self.processor.feature_extractor.pad(
            whisper_inputs,
            padding='longest',
            return_tensors="pt",
        )

        batch['pyannote_inputs'] = [f['pyannote_inputs'] for f in features]
        return batch

def prepare_dataset(batch):
    # process audio
    sample = batch['audio']

    inputs = sample.pop("array", None)
    in_sampling_rate = sample.pop("sampling_rate", None)

    if in_sampling_rate != whisper_sampling_rate:
        inputs = F.resample(torch.from_numpy(np.array(inputs)), in_sampling_rate, whisper_sampling_rate).numpy()
    
    # Whisper inputs: 
    whisper_inputs = feature_extractor(inputs, sampling_rate=whisper_sampling_rate, truncation=False)
    batch['whisper_inputs'] = whisper_inputs

    # Diarization inputs: 
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    batch['pyannote_inputs'] = diarizer_inputs

    return batch


if __name__ == '__main__': 

    # Hyperparameters: 
    per_device_eval_batch_size = 4
    dataloader_num_workers = 1

    audio_column_name = "audio"
    streaming = False

    accelerator = Accelerator()
    device = 'cuda:0'

    # Diarization LM parameters: 
    prompts_options = utils.PromptOptions()
    speaker_prefix = prompts_options.speaker_prefix
    speaker_suffix = prompts_options.speaker_suffix

    # Load the different models: 
    asr_model = "distil-whisper/distil-large-v3"
    diarizer_model = "pyannote/speaker-diarization-3.1"

    asr_processor = WhisperProcessor.from_pretrained(asr_model, token=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(asr_model, token=True)
    attn_implementation = "sdpa" if is_torch_sdpa_available() else "eager"

    asr_model = WhisperForConditionalGeneration.from_pretrained(
        asr_model, 
        token=True, 
        attn_implementation=attn_implementation, 
    )

    whisper_sampling_rate = asr_processor.feature_extractor.sampling_rate

    diarization_pipeline = Pipeline.from_pretrained(diarizer_model)
   
    asr_model, diarization_pipeline = accelerator.prepare(asr_model, diarization_pipeline)

    # Load the dataset: 
    raw_dataset = IterableDatasetDict()

    with accelerator.main_process_first(): 
        if streaming: 
            raw_dataset = load_dataset(
                'kamilakesbi/fisher_medium', 
                streaming=True, 
                num_proc=None,
            )
        else: 
            raw_dataset = Dataset.from_file("/data/fisher/generator/default-f61137895945b655/0.0.0/generator-train-00013-of-00059.arrow").select(range(8))


    with accelerator.main_process_first(): 
        vectorized_dataset = raw_dataset.map(
            prepare_dataset, 
            remove_columns=['audio'],            
        )

    data_collator = DataCollatorWithPadding(
        processor=asr_processor,
        padding="longest",
    )

    dataloader = DataLoader(
            vectorized_dataset,
            batch_size=per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
        )

    for step, batch in enumerate(dataloader):

        print(batch)
        break


        
        # Generate predictions and pad to max generated length
        # generate_fn = model.module.generate if accelerator.num_processes > 1 else model.generate
        # generated_ids = generate_fn(batch["input_features"].to(dtype=torch_dtype), **gen_kwargs)
        # generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id)

        # Gather all predictions and targets
    # dataloader = accelerator.prepare(dataloader)

    # batches = tqdm(dataloader, disable=not accelerator.is_local_main_process)

    # for step, batch in enumerate(dataloader):

    #     print(batch)
    #     break


    # normalizer = WhisperTokenizer.from_pretrained("distil-whisper/distil-large-v3")

    # # Load the preprocessor: 
    # preprocessor = Preprocess(orchestrator, normalizer)

    # dataset = dataset.map(
    #     preprocessor, 
    #     input_columns=['transcripts', 'speakers', 'audio'], 
    #     batched=True, 
    #     batch_size=4,
    #     remove_columns=['transcripts', 'speakers'],  
    #     with_rank=True,
    #     num_proc=20,
    # )

    # print(dataset)








