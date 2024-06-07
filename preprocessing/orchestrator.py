from typing import List, Optional, Union

import numpy as np
import requests
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
from diarizationlm import utils


class OrchestratorPipeline:
    def __init__(
        self,
        asr_pipeline,
        diarization_pipeline,
    ):
        self.asr_pipeline = asr_pipeline
        self.sampling_rate = asr_pipeline.feature_extractor.sampling_rate
        self.diarization_pipeline = diarization_pipeline

        self.prompts_options = utils.PromptOptions()

    @classmethod
    def from_pretrained(
        cls,
        asr_model: Optional[str] = "distil-whisper/distil-large-v3",
        *,
        diarizer_model: Optional[str] = "pyannote/speaker-diarization-3.1",
        chunk_length_s: Optional[int] = 30,
        use_auth_token: Optional[Union[str, bool]] = True,
        **kwargs,
    ):

        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            chunk_length_s=chunk_length_s,
            token=use_auth_token,
            **kwargs,
        )
        diarization_pipeline = Pipeline.from_pretrained(diarizer_model, use_auth_token=use_auth_token)

        return cls(asr_pipeline, diarization_pipeline)
    
    def __call__(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]],
        **kwargs,
    ):
        kwargs_asr = {
            argument[len("asr_") :]: value for argument, value in kwargs.items() if argument.startswith("asr_")
        }

        kwargs_diarization = {
            argument[len("diarization_") :]: value for argument, value in kwargs.items() if argument.startswith("diarization_")
        }
        
        inputs, diarizer_inputs = self.preprocess(inputs)

        print('Diarize: ')
        diarization = self.diarization_pipeline(
            {"waveform": diarizer_inputs, "sample_rate": self.sampling_rate},
            **kwargs_diarization,
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
        
        print('Transcribe: ')
        asr_out = self.asr_pipeline(
            {"array": inputs, "sampling_rate": self.sampling_rate},
            return_timestamps=True, 
            **kwargs_asr,
        )
        transcript_text = asr_out['text'].strip()
        sentences_with_timestamps = asr_out["chunks"]

        word_labels = []
        for sentence_with_timestamps in sentences_with_timestamps: 
            start_timestamp, end_timestamp = sentence_with_timestamps['timestamp']
            sentence = sentence_with_timestamps['text']

            # List of segments that overlap with the current word
            overlap_segments = [segment for segment in new_segments if segment['segment']['end'] >= start_timestamp and segment['segment']['start'] <= end_timestamp]

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
                # If no overlap, associate with closest speaker
                gap_to_end = [float(new_segment['segment']['start'] - end_timestamp) for new_segment in new_segments]
                gap_to_start = [float(new_segment['segment']['end'] - start_timestamp) for new_segment in new_segments]

                gap_end_index = np.argmin(gap_to_end)
                gap_start_index = np.argmin(gap_to_start)

                if gap_to_end[gap_end_index] <= gap_to_start[gap_start_index]: 
                    label = str(int(gap_to_end[gap_end_index]['speaker'][-1]) + 1)
                else: 
                    label = str(int(gap_to_start[gap_start_index]['speaker'][-1]) + 1)

            nb_words_in_sentence = len(sentence.strip().split(' '))

            word_labels += [label]* nb_words_in_sentence

        assert len(word_labels) == len(transcript_text.split(' '))

        word_labels = ' '.join(word_labels)

        return transcript_text, word_labels

    # Adapted from transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline.preprocess
    # (see https://github.com/huggingface/transformers/blob/238449414f88d94ded35e80459bb6412d8ab42cf/src/transformers/pipelines/automatic_speech_recognition.py#L417)
    def preprocess(self, inputs):
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.sampling_rate)

        if isinstance(inputs, dict):
            # Accepting `"array"` which is the key defined in `datasets` for better integration
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = _inputs
            if in_sampling_rate != self.sampling_rate:
                inputs = F.resample(torch.from_numpy(np.array(inputs)), in_sampling_rate, self.sampling_rate).numpy()

        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for ASRDiarizePipeline")

        # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
        diarizer_inputs = torch.from_numpy(inputs).float()
        diarizer_inputs = diarizer_inputs.unsqueeze(0)

        return inputs, diarizer_inputs
    

