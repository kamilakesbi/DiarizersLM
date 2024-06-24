import numpy as np 
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import torch
from torchaudio import functional as F
import psutil
import sys

def vram_monitoring(threhsold = 70):
            
    if psutil.virtual_memory().percent > threhsold: 
        print(psutil.virtual_memory().percent)
        sys.exit(0)


def compute_duration(batch):
        
    batch['duration'] = len(batch['audio']['array']) / batch["audio"]['sampling_rate']

    return batch


def add_batch_to_dataset(
    processed_dataset, 
    ref_diarized_text_batch, 
    ref_text_batch, 
    ref_labels_batch, 
    hyp_text_batch, 
    hyp_labels_batch, 
    hyp_diarized_text_batch, 
    hyp_oracle_labels, 
    hyp_deg_labels
): 
    
    for i in range(len(ref_diarized_text_batch)): 
        dataset_row = {"ref_diarized_text": [], "ref_text": [], "ref_labels": [], "hyp_text": [], "hyp_labels": [], "hyp_diarized_text": [], "hyp_deg_labels": [], "hyp_oracle_labels": []}
        dataset_row['ref_diarized_text'].append(ref_diarized_text_batch[i])
        dataset_row['ref_text'].append(ref_text_batch[i])
        dataset_row['ref_labels'].append(ref_labels_batch[i])
        dataset_row['hyp_text'].append(hyp_text_batch[i])
        dataset_row['hyp_labels'].append(hyp_labels_batch[i])
        dataset_row['hyp_diarized_text'].append(hyp_diarized_text_batch[i])
        dataset_row['hyp_oracle_labels'].append(hyp_oracle_labels[i])
        dataset_row['hyp_deg_labels'].append(hyp_deg_labels[i])
        processed_dataset = processed_dataset.add_item(dataset_row)

    return processed_dataset


@dataclass
class DataCollatorAudio:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor ([`Wav2Vec2FeatureExtractor`])
            The feature extractor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        sampling_rate: 
    """

    processor: Any
    padding: Union[bool, str] = "longest"
    sampling_rate: float = 16000

    def __call__(self, 
        features: List[Dict[str, Union[List[int], np.ndarray]]], 
    ) -> Dict[str, np.ndarray]:

        batch = {}
        samples = [example['audio']["array"] for example in features]       

        in_sampling_rate = features[0]['audio']['sampling_rate']

        if in_sampling_rate != self.sampling_rate:
            samples = [F.resample(torch.from_numpy(np.array(input)), in_sampling_rate, self.sampling_rate).numpy() for input in samples] 

        batch['whisper_inputs'] = self.processor(
            samples,
            sampling_rate=self.sampling_rate,
            truncation=False,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch['pyannote_inputs'] = [torch.from_numpy(sample).float().unsqueeze(0) for sample in samples]
        
        return batch



@dataclass
class DataCollatorLabels:
    """
    """

    def __call__(self, 
        features: List[Dict[str, Union[List[int], np.ndarray]]], 
    ) -> Dict[str, np.ndarray]:

        batch = {}
        batch['timestamps_start'] = [f['timestamps_start'] for f in features]
        batch['timestamps_end'] = [f['timestamps_end'] for f in features]
        batch['speakers'] = [f['speakers'] for f in features]
        batch['transcripts'] = [f['transcripts'] for f in features]
        
        return batch



