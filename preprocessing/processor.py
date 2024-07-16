import numpy as np 
from diarizationlm import utils 


class Processor: 

    def __init__(self,
        diarization_pipeline, 
        asr_model, 
        asr_processor, 
        normalizer, 
        prompts_options,                   
    ): 
        self.diarization_pipeline = diarization_pipeline
        self.asr_model = asr_model
        self.asr_processor = asr_processor
        self.sample_rate = self.asr_processor.feature_extractor.sampling_rate
        self.prompts_options = prompts_options
        self.normalizer = normalizer
        self.speaker_prefix = self.prompts_options.speaker_prefix
        self.speaker_suffix = self.prompts_options.speaker_suffix
 
    def get_diarization_segments(self, diarizer_inputs): 

        diarization_segments = []

        for diarizer_input in diarizer_inputs: 

            diarization = self.diarization_pipeline(
                {"waveform": diarizer_input, "sample_rate": self.sample_rate},
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

    def transcript(self, whisper_inputs): 

        asr_model_out = self.asr_model.generate(**whisper_inputs, return_timestamps=True, return_segments = True)
        transcripts = self.asr_processor.batch_decode(asr_model_out['sequences'], output_offsets=True, skip_special_tokens=True, normalize = True)

        # fix while waiting for https://github.com/huggingface/transformers/pull/32003 to be merged: 
        for i in range(len(transcripts[0]['offsets'])):
            transcripts[0]['offsets'][i]['timestamp'] = (asr_model_out['segments'][0][i]['start'].item(), asr_model_out['segments'][0][i]['end'].item())

        return transcripts


    def orchestrate(self, transcriptions, diarization_segments): 

        transcripts_batch = []
        labels_batch = []
        diarized_transcript_batch = []

        for i, asr_output in enumerate(transcriptions): 

            # transcript_text = asr_output['text'].strip()
            sentences_with_timestamps = asr_output["offsets"]
            diarization_segment = diarization_segments[i]
            word_labels = []

            transcript_text = ''
            for sentence_with_timestamps in sentences_with_timestamps:
                start_timestamp, end_timestamp = sentence_with_timestamps['timestamp']
                sentence = self.normalizer.normalize(sentence_with_timestamps['text']).replace(",", "").replace(".", "").replace("_", "").strip()
                transcript_text += sentence + ' '
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

                nb_words_in_sentence = len(sentence.strip().split())

                word_labels += [label]* nb_words_in_sentence

            if len(word_labels) != len(transcript_text.strip().split()): 
                print('Exception!')
                size = min(len(word_labels), len(transcript_text.strip().split()))
                word_labels = word_labels[:size]
                transcript_text = transcript_text[:size]

            word_labels = ' '.join(word_labels)

            transcripts_batch.append(transcript_text.strip())
            labels_batch.append(word_labels)

        return transcripts_batch, labels_batch

    def get_references(self, ref_transcriptions, ref_speakers): 
        
        ref_texts_batch = []
        ref_labels_batch = []

        for i, transcriptions in enumerate(ref_transcriptions):
            
            # Map speakers to integer values as required by diarizationlm:
            speaker_to_int = {speaker: str(idx + 1) for idx, speaker in enumerate(sorted(set(ref_speakers[i])))}
            speakers = [speaker_to_int[speaker] for speaker in ref_speakers[i]]

            ref_diarized_text = ''
            for index, transcript in enumerate(transcriptions):
                ref_diarized_text += self.speaker_prefix + speakers[index] + self.speaker_suffix + ' '
                ref_diarized_text += self.normalizer.normalize(transcript).replace(",", "").replace(".", "").replace("_", "").strip()
                ref_diarized_text += ' '

            ref_text, ref_labels = utils.extract_text_and_spk(ref_diarized_text, po=self.prompts_options)

            ref_texts_batch.append(ref_text)
            ref_labels_batch.append(ref_labels)

        return ref_texts_batch, ref_labels_batch

    def get_oracle_and_degraded_speakers(self, hyp_text_batch, hyp_labels_batch, ref_text_batch, ref_labels_batch): 

        deg_speakers = []
        oracle_speakers = []
        for i in range(len(hyp_text_batch)): 
            
            try: 
                oracle_speakers.append(utils.transcript_preserving_speaker_transfer(
                            src_text=ref_text_batch[i],
                            src_spk=ref_labels_batch[i],
                            tgt_text=hyp_text_batch[i],
                            tgt_spk=hyp_labels_batch[i],
                ))
            except: 
                print('exception')
                deg_speakers.append('') 
            
            try: 
                deg_speakers.append(utils.transcript_preserving_speaker_transfer(
                            src_text=hyp_text_batch[i],
                            src_spk=hyp_labels_batch[i],
                            tgt_text=ref_text_batch[i],
                            tgt_spk=ref_labels_batch[i],
                ))
            except:
                print('exception')
                deg_speakers.append('')
        
        return oracle_speakers, deg_speakers
    

def add_oracle_and_deg_labels(batch):

    try: 
        batch['ref_spk_degraded'] = [utils.transcript_preserving_speaker_transfer(
                        src_text=batch['hyp_text'][0],
                        src_spk=batch['hyp_spk'][0],
                        tgt_text=batch['ref_text'][0],
                        tgt_spk=batch['ref_spk'][0],
                    )]
    except: 
        print('exception')
        batch['ref_spk_degraded'] = ['']
        
    try: 
        batch['hyp_spk_oracle'] = [utils.transcript_preserving_speaker_transfer(
                        src_text=batch['ref_text'][0],
                        src_spk=batch['ref_spk'][0],
                        tgt_text=batch['hyp_text'][0],
                        tgt_spk=batch['hyp_spk'][0],
                    )]
    except: 
        print('exception')
        batch['hyp_spk_oracle'] = ['']

    return batch