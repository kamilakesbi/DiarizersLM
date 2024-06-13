from datasets import load_dataset, Dataset
from diarizationlm import utils
import torch 
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer
from pyannote.audio import Pipeline

from transformers.utils import is_torch_sdpa_available 
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils import add_batch_to_dataset, DataCollatorAudio, DataCollatorLabels
from processor import Processor
from tqdm import tqdm 
import logging 
import time


if __name__ == '__main__': 

    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('my_log_file.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Hyperparameters: 
    dataset_name = 'kamilakesbi/fisher_medium'
    split = "train"
    dataloader_batch_size = 16
    dataloader_num_workers = 8
    preprocessing_num_workers = 24
    streaming = True

    logger.debug('batch size: {}'.format(dataloader_batch_size))
    logger.debug('Data loader num workers: {}'.format(dataloader_num_workers))
    logger.debug('Preprocessing num workers: {}'.format(preprocessing_num_workers))

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
                dataset_name, 
                split='train', 
                streaming=True, 
                num_proc=None,
            )
        else: 
            raw_dataset = load_dataset(
                dataset_name, 
                split='train', 
                streaming=False, 
                num_proc=preprocessing_num_workers,
            )

    label_dataset = raw_dataset.select_columns(['timestamps_start', 'timestamps_end', 'speakers', 'transcripts'])
    audio_dataset = raw_dataset.select_columns(['audio'])

    # Define Data Collators: 
    audio_data_collator = DataCollatorAudio(
        processor=asr_processor,
        padding="longest",
        sampling_rate=sample_rate
    )

    labels_data_collator = DataCollatorLabels()

    # Define Data Loaders: 
    audio_dataloader = DataLoader(
            audio_dataset,
            batch_size=dataloader_batch_size,
            collate_fn=audio_data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
        )
    
    labels_dataloader = DataLoader(
        label_dataset,
        batch_size=dataloader_batch_size * accelerator.num_processes,
        collate_fn=labels_data_collator,
        num_workers=dataloader_num_workers,
    )

    audio_dataloader = accelerator.prepare(audio_dataloader)
    audio_batches = tqdm(audio_dataloader, disable=not accelerator.is_local_main_process)

    processed_dataset = Dataset.from_dict({"ref_diarized_text": [], "ref_text": [], "ref_labels": [], "hyp_text": [], "ref_labels": []})

    print('Entering dataloder loop: ')

    start_time = time.perf_counter()
    for step, (audio_batch, labels_batch) in tqdm(enumerate(zip(audio_dataloader,labels_dataloader))):
        
        logger.debug('Data loading time: {}'.format(time.perf_counter() - start_time))

        # Diarization: 
        start_time = time.perf_counter()
        diarizer_inputs = audio_batch['pyannote_inputs']
        diarization_segments = processor.get_diarization_segments(diarizer_inputs)
        logger.debug('Diarization time: {}'.format(time.perf_counter() - start_time))

        # Transcription: 
        start_time = time.perf_counter()

        whisper_inputs = audio_batch['whisper_inputs']
        whisper_inputs.input_features = whisper_inputs.to(device)
        transcriptions = processor.transcript(whisper_inputs)

        logger.debug('Transcription: {}'.format(time.perf_counter() - start_time))

        # Orchestration: 
        start_time = time.perf_counter()

        hyp_text_batch, hyp_labels_batch, hyp_diarized_text_batch = processor.orchestrate(transcriptions, diarization_segments)
        ref_text_batch, ref_labels_batch, ref_diarized_text_batch = processor.get_references(labels_batch['transcripts'], labels_batch['speakers'])
        
        logger.debug('Orchestration : {}'.format(time.perf_counter() - start_time))

        processed_dataset = add_batch_to_dataset(
            processed_dataset, 
            ref_diarized_text_batch, 
            ref_text_batch, 
            ref_labels_batch, 
            hyp_text_batch, 
            hyp_labels_batch, 
            hyp_diarized_text_batch 
        )
        start_time = time.perf_counter()

        accelerator.wait_for_everyone()

    processed_dataset.push_to_hub('kamilakesbi/test', private=True)






