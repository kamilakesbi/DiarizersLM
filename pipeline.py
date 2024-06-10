from typing import List, Optional, Union

import numpy as np
import requests
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
from diarizationlm import utils
import diarizationlm


class DiarizersLmPipeline:
    def __init__(
        self,
        asr_pipeline,
        diarization_pipeline,
        llm_pipeline, 
    ):
        self.asr_pipeline = asr_pipeline
        self.sampling_rate = asr_pipeline.feature_extractor.sampling_rate
        self.diarization_pipeline = diarization_pipeline

        self.prompts_options = utils.PromptOptions()
        self.llm_pipeline = llm_pipeline
        self.terminators = [
            llm_pipeline.tokenizer.eos_token_id,
            llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.max_new_tokens = 4096

    @classmethod
    def from_pretrained(
        cls,
        asr_model: Optional[str] = "distil-whisper/distil-large-v3",
        *,
        diarizer_model: Optional[str] = "pyannote/speaker-diarization-3.1",
        llm_model: Optional[str] = "meta-llama/Meta-Llama-3-8B",
        chunk_length_s: Optional[int] = 30,
        use_auth_token: Optional[Union[str, bool]] = True,
        attn_implementation: Optional[str] = None, 
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
        if 'device' in kwargs: 
            diarization_pipeline.to(torch.device(kwargs['device']))
        
        llm_model = pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16, 'attn_implementation': attn_implementation},
            token=use_auth_token,
            **kwargs,
        )

        return cls(asr_pipeline, diarization_pipeline, llm_model)
    
    def __call__( 
            self,        
            inputs: Union[np.ndarray, List[np.ndarray]],
            **kwargs,
        ): 

        print('Diarize and Transcribe: ')
        hyp_text, hyp_labels = self.orchestrate(inputs, **kwargs)

        print('Generate prompts: ')
        prompts = self.generate_prompts(hyp_text, hyp_labels)

        print('Generate completions: ')
        completions = self.generate_completions(prompts)
        
        print('Post process completions: ')
        output = self.post_process(completions, hyp_text, hyp_labels)

        return output

    def orchestrate(
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

    def generate_prompts(
        self, 
        text, 
        labels, 
    ): 
        
        utterance = {"utterance_id": "0",  "hyp_text": str(text) , "hyp_spk": labels}
        prompts = diarizationlm.generate_prompts(utterance, self.prompts_options)

        return prompts

    def generate_completions(
        self, 
        prompts
    ): 
        completions = []
        for prompt in prompts: 

            message = [
                {"role": "system", "content": "In the speaker diarization transcript below, some words are potentially misplaced."
                    " Please correct those words and move them to the right speaker. For example, given this input transcript: "
                    " <spk:1> How are you doing today? I <spk:2> am doing very well. How was everything at the <spk:1> party? Oh, the party? It was awesome. We had lots of fun. Good <spk:2> to hear!"
                    "The correct output transcript should be:"
                    "<spk:1> How are you doing today? <spk:2> I am doing very well. How was everything at the party? <spk:1> Oh, the party? It was awesome. We had lots of fun. <spk:2> Good to hear!"  
                    " Now, please correct the transcript below. Give only the corrected transcript, without additional comments."
                },
                {"role": "user", "content": prompt},
            ]

            output = self.llm_pipeline(
                message, 
                eos_token_id=self.terminators, 
                max_new_tokens=self.max_new_tokens,
            )

            completions.append(output[0]["generated_text"][-1])

        return completions
    
    def post_process(
        self, 
        completions, 
        hyp_text, 
        hyp_labels, 
    ): 

        completions_list = []
        

        for completion in completions: 

            completion = completion['content']

            if self.prompts_options.completion_suffix and self.prompts_options.completion_suffix in completion:
                completion = utils.truncate_suffix_and_tailing_text(
                    completion, self.prompts_options.completion_suffix
                )
            completions_list.append(completion)
        completions = " ".join(completions_list).strip()

        llm_text, llm_labels = utils.extract_text_and_spk(
            completions, po=self.prompts_options
        )

        transferred_llm_labels = utils.transcript_preserving_speaker_transfer(
            src_text=llm_text,
            src_spk=llm_labels,
            tgt_text=hyp_text,
            tgt_spk=hyp_labels,
        )

        return utils.create_diarized_text(hyp_text.split(' '), transferred_llm_labels.split(' '))

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
                inputs = F.resample(torch.from_numpy(inputs), in_sampling_rate, self.sampling_rate).numpy()

        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for ASRDiarizePipeline")

        # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
        diarizer_inputs = torch.from_numpy(inputs).float()
        diarizer_inputs = diarizer_inputs.unsqueeze(0)

        return inputs, diarizer_inputs
    


