from typing import Any
from datasets import Dataset
import re 
from diarizationlm import utils
from orchestrator import OrchestratorPipeline
import torch 
from multiprocess import set_start_method

class Preprocess: 

    def __init__(
        self,
        orchestrator,
    ) -> None:
        
        set_start_method('spawn')

        self.junk_tokens =  ["[noise]", "[laughter]", "[silence]", "[vocalized-noise]", "<a_aside>", "<b_aside>", "<e_aside>",
            "[laughter-", "_1", "[laugh]", "[sigh]", "[cough]", "[mn]", "[breath]", "[lipsmack]",
            "[sneeze]", "[skip]", "[pause]", "(%hesitation)", "(%HESITATION)"]
        
        self.fisher_punctuations = ["{", "}", "[", "]-", "]", "((", "))", "(", ")", "."]
        self.fisher_fillers = r"\b(uh|uhm|um|hmm|mm|mhm|mmm)\b"

        self.prompt_options = utils.PromptOptions()
        self.speaker_prefix = self.prompt_options.speaker_prefix
        self.speaker_suffix = self.prompt_options.speaker_suffix

        self.orchestrator = orchestrator

    def preprocess_text(self, text): 

        for disfluency in self.junk_tokens: 
            text = text.replace(disfluency, '')

        # normalise acronyms (Fisher: u_.c_.l_.a., SWBD: u c l a)
        text = text.replace("_.", " ").replace(".", "")

        # normalise acronyms (Fisher: u_.c_.l_.a., SWBD: u c l a)
        text = text.replace("_.", " ").replace(".", "")
        # Replace partially pronounced words (square brackets + hyphen): westmin[ster]- to westmin- or -[go]ing to -ing
        # Replace anomalous words (square brackets + backslack): [lemguini/linguini] to linguini
        # Replace the combo of the two: [lem[guini]-/linguini] to lem-
        # Example: we [ah/are] -[go]ing to westmin[ster]- for [lem[guini]-/linguini]
        # Target: we ah -ing to westmin- for lem-
        # Treat anomalous words first then destroy the content of all square brackets (partially pronounced words)

        # First treat partially pronounced anomalous words by removing correct word: [lem[guini]-/linguini] to [lem[guini]-
        text = re.sub(r"\-\/.*?\]", "-", text)

        # Now replace anomalous words with their correct transcriptions: [lemguini/linguini] to linguini
        split_str = text.split("/")
        if len(split_str) > 1:
            text = " ".join(
                [" ".join([" ".join(i.split(" ")[:-1]) for i in split_str])] + [split_str[-1].split(" ")[-1]])
          
        # Remove the trailing brackets on the start/end of words
        processed_str = []
        for word in text.split():
            if word[0] == "[":
                processed_str.append(word[1:])
            elif word[-1] == "]":
                processed_str.append(word[:-1])
            else:
                processed_str.append(word)
        
        # Stick the processed words back together
        text = " ".join(processed_str)

        # Now we can remove all words in square brackets: -[go]ing to -ing
        text = re.sub(r"\-\[(.*?)\]", "-", text)
        
        # westmin[ster]- to westmin-
        text = re.sub(r"\[(.*?)\]\-", "-", text)

        # tech[n]ology to tech-ology
        text = re.sub(r"\[(.*?)\]", "-", text)

        # partially pronounced words are now done!
        # remove erroneous punctuations (curly braces, trailing square brackets, etc.)
        for punctuation in self.fisher_punctuations:
            text = text.replace(punctuation, "")

        # Remove fillers from the train set not present in the test set
        text = re.sub(self.fisher_fillers, "", text)

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
        for i, audio in enumerate(audio_column):
            
            # Map speakers to integer values as required by diarizationlm:
            speaker_to_int = {speaker: str(idx + 1) for idx, speaker in enumerate(sorted(set(speakers_column[i])))}
            speakers = [speaker_to_int[speaker] for speaker in speakers_column[i]]

            transcriptions = transcripts_column[i]

            for index, transcript in enumerate(transcriptions):
                ref_diarized_text += self.speaker_prefix + speakers[index] + self.speaker_suffix + ' '
                ref_diarized_text += self.preprocess_text(transcript)
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

    dataset = Dataset.from_file("/raid/kamilakesbi/generator/default-0af89f8814d3d2f4/0.0.0/generator-train-00000-of-00040.arrow")
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    orchestrator = OrchestratorPipeline.from_pretrained(
        asr_model = "openai/whisper-large-v3",
        diarizer_model = "pyannote/speaker-diarization-3.1", 
    )
    # Load the preprocessor: 
    preprocessor = Preprocess(orchestrator)

    dataset = dataset.map(
        preprocessor, 
        input_columns=['transcripts', 'speakers', 'audio'], 
        batched=True, 
        batch_size=8,
        remove_columns=['transcripts', 'speakers'],  
        with_rank=True,
        num_proc=2,
    )

    print(dataset)








