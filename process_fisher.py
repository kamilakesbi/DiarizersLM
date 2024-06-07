from typing import Any
from datasets import Dataset, Audio
import re 
from transformers import WhisperTokenizer
from diarizationlm import utils

class Preprocess: 

    def __init__(
        self, 
        whisper_model = "distil-whisper/distil-large-v3", 
    ) -> None:
        
        # Load the Whisper tokenizer
        tokenizer = WhisperTokenizer.from_pretrained(whisper_model)
        self.vocab = tokenizer.get_vocab()

        self.junk_tokens =  ["[noise]", "[laughter]", "[silence]", "[vocalized-noise]", "<a_aside>", "<b_aside>", "<e_aside>",
            "[laughter-", "_1", "[laugh]", "[sigh]", "[cough]", "[mn]", "[breath]", "[lipsmack]",
            "[sneeze]", "[skip]", "[pause]", "(%hesitation)", "(%HESITATION)"]
        
        self.fisher_punctuations = ["{", "}", "[", "]-", "]", "((", "))", "(", ")", "."]
        self.fisher_fillers = r"\b(uh|uhm|um|hmm|mm|mhm|mmm)\b"

        self.prompt_options = utils.PromptOptions()

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
    
    def __call__(self, transcripts_column, speakers_column):

        new_batch = {"ref_diarized_text": [], 'ref_text': [], 'ref_labels': []} 

        # batch = [{key: values[i] for key, values in files.items()} for i in range(len(files["transcripts"]))]

        speaker_prefix = "<speaker:"
        speaker_suffix = '>'
        ref_diarized_text = ''
        for i, speakers in enumerate(speakers_column): 
            
            # Map speakers to integer values as required by diarizationlm: 
            speaker_to_int = {speaker: str(idx + 1) for idx, speaker in enumerate(sorted(set(speakers)))}
            speakers = [speaker_to_int[speaker] for speaker in speakers]

            transcriptions = transcripts_column[i]

            for index, transcript in enumerate(transcriptions): 
                ref_diarized_text += speaker_prefix + speakers[index] + speaker_suffix + ' '
                ref_diarized_text += self.preprocess_text(transcript)
                ref_diarized_text += ' '

            ref_text, ref_labels = utils.extract_text_and_spk(ref_diarized_text, po=self.prompt_options)

            new_batch['ref_diarized_text'].append(ref_diarized_text)
            new_batch['ref_text'].append(ref_text)
            new_batch['ref_labels'].append(ref_labels)

        return new_batch


if __name__ == '__main__': 

    dataset = Dataset.from_file("/raid/kamilakesbi/generator/default-0af89f8814d3d2f4/0.0.0/generator-train-00000-of-00040.arrow")

    # Load the preprocessor: 
    preprocessor = Preprocess()

    dataset = dataset.map(
        preprocessor, 
        input_columns=['transcripts', 'speakers'], 
        batched=True, 
        batch_size=32,
        remove_columns=['transcripts', 'speakers'],  
        num_proc=12, 
    ).cast_column('audio', Audio())

    print(dataset)










