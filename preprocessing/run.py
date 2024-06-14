from datasets import load_dataset, Dataset
from diarizationlm import utils
import torch 
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer
from pyannote.audio import Pipeline

from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from utils import DataCollatorAudio, DataCollatorLabels, add_batch_to_dataset
from processor import Processor
from tqdm import tqdm 
import logging 
import time
from datetime import timedelta

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """
    asr_name_or_path: str = field(
        metadata={"help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"}
    )
    diarizer_name_or_path: str = field(
        metadata={"help": "Path to pretrained Pyannote model or model identifier from huggingface.co/models"}
    )

    normalizer_name_or_path: Optional[str] = field(
        default = None, 
        metadata={"help": "Path to pretrained noramlizer model or model identifier from huggingface.co/models"}
    )

    attn_implementation: Optional[str] = field(
            default=None,
            metadata={
                "help": (
                    "Which attention implementation to use in the encoder and decoder attention layers. Can be one of:\n"
                    "1. `eager` or `None`: default Transformers attention implementation.\n"
                    "2. `sdpa`: Flash Attention through PyTorch SDPA. Requires `torch>=2.1`. Recommended for hardware where Flash Attention 2 is not supported, e.g. Turing GPUs, (T4, RTX 2080).\n"
                    "3. `flash_attn_2`: Flash Attention 2 through the Flash Attention package https://github.com/Dao-AILab/flash-attention. **Always** recommended on supported hardware (Ampere, Ada, or Hopper GPUs, e.g., A100, RTX 3090, RTX 4090, H100)."
                )
            },
        )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to load the model weights. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )

    
@dataclass
class DataArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    dataset_split_name: str = field(
        default="train",
        metadata={
            "help": (
                "The name of the data set splits to use (via the datasets library)."
                " Defaults to 'train'. Multiple splits can be passed by splitting a"
                " list through the '+' character, e.g. 'train+validation' will"
                " pseudo-label both the 'train' and 'validation' splits sequentially."
            )
        },
    )

    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use dataset's streaming mode to load and pre-process the data."},
    )

    per_device_batch_size: int = field(
        default=None,
        metadata={"help": "Per device batch size used by the dataloader."},
    )

    dataloader_num_workers: int = field(
        default=None,
        metadata={"help": "Number of workers used in the Dataloader"},
    )

    num_proc: int = field(
        default=None,
        metadata={"help": "Number of workers used to load the dataset"},
    )

    push_to_hub: bool = field(
        default=True,
        metadata={"help": "Whether to push the processed dataset to the Hub or not."},
    )

    output_hub_repository: str= field(
        default=None,
        metadata={"help": "Hub repository"},
    )

    log_file_name: str = field(
        default = None, 
        metadata={"help": "Log File Name"},
    )

if __name__ == '__main__': 

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(str(data_args.log_file_name))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    # Dataset Processing Hyperparameters : 
    dataset_name = str(data_args.dataset_name)
    dataset_split_name = str(data_args.dataset_split_name)
    per_device_batch_size = int(data_args.per_device_batch_size)
    dataloader_num_workers = int(data_args.dataloader_num_workers)
    num_proc = int(data_args.num_proc) if data_args.num_proc is not None else None
    streaming = str(data_args.streaming)

    logger.debug('Per device batch size: {}'.format(per_device_batch_size))
    logger.debug('Data loader num workers: {}'.format(dataloader_num_workers))
    if not streaming: 
        logger.debug('Dataset loading num workers: {}'.format(dataloader_num_workers))

    # Load the different models: 
    asr_model_name = str(model_args.asr_name_or_path)
    diarizer_model_name = str(model_args.diarizer_name_or_path)
    normalizer_name = str(model_args.normalizer_name_or_path) if model_args.normalizer_name_or_path else asr_model_name

    if model_args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif model_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[kwargs],
     
    )
    device = accelerator.device

    device = 'cuda:0'

    # Load prompt options: 
    prompts_options = utils.PromptOptions()

    # Load ASR Model: 
    asr_processor = WhisperProcessor.from_pretrained(asr_model_name, token=True)

    asr_model = WhisperForConditionalGeneration.from_pretrained(
        asr_model_name, 
        token=True, 
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        attn_implementation=str(model_args.attn_implementation), 
    ).to(torch.device(device))

    # Load diarization pipeline: 
    diarization_pipeline = Pipeline.from_pretrained(diarizer_model_name).to(torch.device(device))

    # Load Normalizer: 
    normalizer = WhisperTokenizer.from_pretrained(str(normalizer_name))

    sample_rate = asr_processor.feature_extractor.sampling_rate

    # Prepare models for accelerate: 
    diarization_pipeline, asr_model, asr_processor, normalizer = accelerator.prepare(diarization_pipeline, asr_model, asr_processor, normalizer)

    # Prepare Processor: 
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
                split=dataset_split_name, 
                streaming=True, 
                num_proc=None,
            )
        else: 
            raw_dataset = load_dataset(
                dataset_name, 
                split=dataset_split_name, 
                streaming=False, 
                num_proc=num_proc,
            )

    accelerator.wait_for_everyone()

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
            batch_size=per_device_batch_size,
            collate_fn=audio_data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
        )
    
    labels_dataloader = DataLoader(
        label_dataset,
        batch_size= per_device_batch_size * accelerator.num_processes,
        collate_fn=labels_data_collator,
        num_workers=dataloader_num_workers,
    )

    audio_dataloader = accelerator.prepare(audio_dataloader)
    audio_batches = tqdm(audio_dataloader, disable=not accelerator.is_local_main_process)

    processed_dataset = Dataset.from_dict({"ref_diarized_text": [], "ref_text": [], "ref_labels": [], "hyp_text": [], "ref_labels": []})

    logger.debug('Entering dataloder loop: ')


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
        
        break


    if accelerator.is_main_process:
        if str(data_args.push_to_hub): 
            processed_dataset.push_to_hub(str(data_args.output_hub_repository), private=True)






