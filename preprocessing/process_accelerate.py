from datasets import load_dataset, IterableDatasetDict, Dataset, concatenate_datasets
from diarizationlm import utils
import torch 
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
from pyannote.audio import Pipeline

from transformers.utils import is_torch_sdpa_available 
from accelerate import Accelerator
from torchaudio import functional as F
import numpy as np 
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union
from dataclasses import dataclass


def get_diarization_segments(diarizer_inputs, diarization_pipeline): 

    diarization_segments = []

    for diarizer_input in diarizer_inputs: 

        diarization = diarization_pipeline(
            {"waveform": diarizer_input, "sample_rate": whisper_sampling_rate},
        )

        segments = []
        for segment, track, label in diarization.itertracks(yield_label=True):
            segments.append({'segment': {'start': segment.start, 'end': segment.end},
                            'track': track,
                            'label': label})

        # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
        # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
        new_segments = []
        prev_segment = cur_segment = segments[0]

        for i in range(1, len(segments)):
            cur_segment = segments[i]

            # check if we have changed speaker ("label")
            if cur_segment["label"] != prev_segment["label"] and i < len(segments):
                # add the start/end times for the super-segment to the new list
                new_segments.append(
                    {
                        "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["start"]},
                        "speaker": prev_segment["label"],
                    }
                )
                prev_segment = segments[i]

        # add the last segment(s) if there was no speaker change
        new_segments.append(
            {
                "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["end"]},
                "speaker": prev_segment["label"],
            }
        )
        diarization_segments.append(new_segments)

    return diarization_segments

def transcript(whisper_inputs): 

    asr_model_out = asr_model.generate(**whisper_inputs, return_timestamps=True)
    transcripts = asr_processor.batch_decode(asr_model_out, output_offsets=True, skip_special_tokens=True)

    return transcripts


def orchestrate(transcriptions, diarization_segments): 

    transcripts_batch = []
    labels_batch = []
    diarized_transcript_batch = []

    for i, asr_output in enumerate(transcriptions): 

        transcript_text = asr_output['text'].strip()
        sentences_with_timestamps = asr_output["offsets"]
        diarization_segment = diarization_segments[i]
        word_labels = []

        for sentence_with_timestamps in sentences_with_timestamps: 
            start_timestamp, end_timestamp = sentence_with_timestamps['timestamp']
            sentence = sentence_with_timestamps['text']

            # List of segments that overlap with the current word
            overlap_segments = [segment for segment in diarization_segment if segment['segment']['end'] >= start_timestamp and segment['segment']['start'] <= end_timestamp]

            if len(overlap_segments) > 0: 
                # Get segment which has highest overlap with current word
                max_overlap = 0
                current_index=0
                for index, segment in enumerate(overlap_segments):
                    sentence_segment_overlap = min(segment['segment']['end'], end_timestamp) - max(segment['segment']['start'], start_timestamp)
                    if sentence_segment_overlap >= max_overlap:
                        current_index = index
                        max_overlap = sentence_segment_overlap
                        
                label = str(int(overlap_segments[current_index]['speaker'][-1]) + 1)

            else:
                # If no overlap, associate with closest speaker:
                gap_to_end = [float(new_segment['segment']['start'] - end_timestamp) for new_segment in diarization_segment]
                gap_to_start = [float(new_segment['segment']['end'] - start_timestamp) for new_segment in diarization_segment]

                gap_end_index = np.argmin(gap_to_end)
                gap_start_index = np.argmin(gap_to_start)

                if gap_to_end[gap_end_index] <= gap_to_start[gap_start_index]: 
                    label = str(int(diarization_segment[gap_end_index]['speaker'][-1]) + 1)
                else: 
                    label = str(int(diarization_segment[gap_start_index]['speaker'][-1]) + 1)

            nb_words_in_sentence = len(sentence.strip().split(' '))

            word_labels += [label]* nb_words_in_sentence

        assert len(word_labels) == len(transcript_text.split(' '))

        word_labels = ' '.join(word_labels)

        transcripts_batch.append(transcript_text)
        labels_batch.append(word_labels)

    for i in range(len(transcripts_batch)): 
        diarized_transcript_batch.append(utils.create_diarized_text(transcripts_batch[i].split(' '), transcripts_batch[i].split(' ')))

    return transcripts_batch, labels_batch, diarized_transcript_batch

def get_references(ref_transcriptions, ref_speakers): 
    
    ref_texts_batch = []
    ref_labels_batch = []
    ref_diarized_texts_batch = []

    for i, transcriptions in enumerate(ref_transcriptions):
        
        # Map speakers to integer values as required by diarizationlm:
        speaker_to_int = {speaker: str(idx + 1) for idx, speaker in enumerate(sorted(set(ref_speakers[i])))}
        speakers = [speaker_to_int[speaker] for speaker in ref_speakers[i]]

        ref_diarized_text = ''
        for index, transcript in enumerate(transcriptions):
            ref_diarized_text += speaker_prefix + speakers[index] + speaker_suffix + ' '
            ref_diarized_text += normalizer.normalize(transcript)
            ref_diarized_text += ' '

        ref_text, ref_labels = utils.extract_text_and_spk(ref_diarized_text, po=prompts_options)

        ref_texts_batch.append(ref_text)
        ref_labels_batch.append(ref_labels)
        ref_diarized_texts_batch.append(ref_diarized_text)

    return ref_texts_batch, ref_labels_batch, ref_diarized_texts_batch

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

        samples = [example[audio_column_name]["array"] for example in features]

        in_sampling_rate = features[0]['audio']['sampling_rate']

        if in_sampling_rate != whisper_sampling_rate:
            samples = [F.resample(torch.from_numpy(np.array(input)), in_sampling_rate, whisper_sampling_rate).numpy() for input in samples] 

        batch['whisper_inputs'] = feature_extractor(
            samples,
            sampling_rate=whisper_sampling_rate,
            truncation=False,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch['pyannote_inputs'] = [torch.from_numpy(sample).float().unsqueeze(0) for sample in samples]
        
        return batch

def add_batch_to_dataset(
    processed_dataset, 
    ref_diarized_text_batch, 
    ref_text_batch, 
    ref_labels_batch, 
    hyp_text_batch, 
    hyp_labels_batch, 
): 
    
    for i in range(len(ref_diarized_text_batch)): 
        dataset_row = {"ref_diarized_text": [], "ref_text": [], "ref_labels": [], "hyp_text": [], "hyp_labels": [], "hyp_diarized_text": []}
        dataset_row['ref_diarized_text'].append(ref_diarized_text_batch[i])
        dataset_row['ref_text'].append(ref_text_batch[i])
        dataset_row['ref_labels'].append(ref_labels_batch[i])
        dataset_row['hyp_text'].append(hyp_text_batch[i])
        dataset_row['hyp_labels'].append(hyp_labels_batch[i])
        dataset_row['hyp_diarized_text'].append(hyp_diarized_text_batch[i])
        processed_dataset = processed_dataset.add_item(dataset_row)

    return processed_dataset


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

    normalizer = WhisperTokenizer.from_pretrained("distil-whisper/distil-large-v3")

    whisper_sampling_rate = asr_processor.feature_extractor.sampling_rate

    diarization_pipeline = Pipeline.from_pretrained(diarizer_model).to(torch.device(device))
   
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
            raw_dataset = Dataset.from_file("/data/fisher/generator/default-f61137895945b655/0.0.0/generator-train-00013-of-00059.arrow").select(range(4))


    data_collator = DataCollatorWithPadding(
        processor=asr_processor,
        padding="longest",
    )

    dataloader = DataLoader(
            raw_dataset,
            batch_size=per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
        )

    processed_dataset = Dataset.from_dict({"ref_diarized_text": [], "ref_text": [], "ref_labels": [], "hyp_text": [], "ref_labels": []})

    for step, batch in enumerate(dataloader):
        
        diarizer_inputs = batch['pyannote_inputs']
        
        diarization_segments = get_diarization_segments(diarizer_inputs, diarization_pipeline)

        whisper_inputs = batch['whisper_inputs']
        whisper_inputs.input_features = whisper_inputs.to(device)

        transcriptions = transcript(whisper_inputs)
        
        hyp_text_batch, hyp_labels_batch, hyp_diarized_text_batch = orchestrate(transcriptions, diarization_segments)
        ref_text_batch, ref_labels_batch, ref_diarized_text_batch = get_references(batch['transcripts'], batch['speakers'])

        processed_dataset = add_batch_to_dataset(
            processed_dataset, 
            ref_diarized_text_batch, 
            ref_text_batch, 
            ref_labels_batch, 
            hyp_text_batch, 
            hyp_labels_batch
        )

        print( 'ok')

        










