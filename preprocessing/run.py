from datasets import load_dataset, Dataset
from diarizationlm import utils
import torch 
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer
from pyannote.audio import Pipeline

from transformers.utils import is_torch_sdpa_available 
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils import add_batch_to_dataset, DataCollatorWithPadding
from processor import Processor
from tqdm import tqdm 


if __name__ == '__main__': 

    # Hyperparameters: 
    dataset_name = 'kamilakesbi/fisher_medium'
    split = "train"
    dataloader_batch_size = 8
    dataloader_num_workers = 24
    preprocessing_num_workers = 24
    streaming = True

    # Load the different models: 
    asr_model = "distil-whisper/distil-large-v3"
    diarizer_model = "pyannote/speaker-diarization-3.1"

    accelerator = Accelerator()
    device = accelerator.device

    # Diarization LM parameters: 
    prompts_options = utils.PromptOptions()
   

    asr_processor = WhisperProcessor.from_pretrained(asr_model, token=True)

    attn_implementation = "sdpa" if is_torch_sdpa_available() else "eager"
    asr_model = WhisperForConditionalGeneration.from_pretrained(
        asr_model, 
        token=True, 
        attn_implementation=attn_implementation, 
    )

    normalizer = WhisperTokenizer.from_pretrained("distil-whisper/distil-large-v3")

    diarization_pipeline = Pipeline.from_pretrained(diarizer_model).to(torch.device(device))

    sample_rate = asr_processor.feature_extractor.sampling_rate

    diarization_pipeline, asr_model, asr_processor, normalizer = accelerator.prepare(diarization_pipeline, asr_model, asr_processor, normalizer)

    processor = Processor(
        diarization_pipeline, 
        asr_model, 
        asr_processor, 
        normalizer, 
        prompts_options
    )

    with accelerator.main_process_first(): 
        if streaming: 
            raw_dataset = load_dataset(
                'kamilakesbi/fisher_medium', 
                split='train', 
                streaming=True, 
                num_proc=None,
            )
        else: 
            raw_dataset = load_dataset(
                'kamilakesbi/fisher_medium', 
                split='train', 
                streaming=False, 
                num_proc=preprocessing_num_workers,
            )


    data_collator = DataCollatorWithPadding(
        processor=asr_processor,
        padding="longest",
        sampling_rate=sample_rate
    )

    dataloader = DataLoader(
            raw_dataset,
            batch_size=dataloader_batch_size,
            collate_fn=data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
        )

    dataloader = accelerator.prepare(dataloader)

    processed_dataset = Dataset.from_dict({"ref_diarized_text": [], "ref_text": [], "ref_labels": [], "hyp_text": [], "ref_labels": []})

    for step, batch in tqdm(enumerate(dataloader)):
        
        print('we are here: ')
        diarizer_inputs = batch['pyannote_inputs']
        
        diarization_segments = processor.get_diarization_segments(diarizer_inputs)

        print('now here: ')
        whisper_inputs = batch['whisper_inputs']
        whisper_inputs.input_features = whisper_inputs.to(device)

        print('here: ')
        transcriptions = processor.transcript(whisper_inputs)
        
        hyp_text_batch, hyp_labels_batch, hyp_diarized_text_batch = processor.orchestrate(transcriptions, diarization_segments)
        ref_text_batch, ref_labels_batch, ref_diarized_text_batch = processor.get_references(batch['transcripts'], batch['speakers'])

        processed_dataset = add_batch_to_dataset(
            processed_dataset, 
            ref_diarized_text_batch, 
            ref_text_batch, 
            ref_labels_batch, 
            hyp_text_batch, 
            hyp_labels_batch
        )

    processed_dataset.push_to_hub('kamilakesbi/test', private=True)










