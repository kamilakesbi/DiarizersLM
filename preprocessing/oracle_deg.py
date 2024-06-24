from datasets import load_dataset

from diarizationlm import utils
from transformers import  WhisperTokenizer

def normalize_diarized_text(diarized_text): 

    hyp_text, hyp_labels = utils.extract_text_and_spk(diarized_text, prompts_options)

    hyp_text = hyp_text.split(' ')
    hyp_labels = hyp_labels.split(' ')

    uncased_text = []
    uncased_labels = []
    for i in range(len(hyp_text)): 
        hyp_text_norm = normalizer.normalize(hyp_text[i])
        uncased_text += hyp_text_norm.split(' ')
        uncased_labels += [hyp_labels[i]]*len(hyp_text_norm.split(' '))

    hyp_norm_text, hyp_norm_labels = ' '.join(uncased_text), ' '.join(uncased_labels)

    norm_diarized_text = utils.create_diarized_text(hyp_norm_text.split(' '), hyp_norm_labels.split(' '))

    return norm_diarized_text


def add_oracle_and_deg_labels(batch):

    try: 
        batch['hyp_norm_deg_labels'] = utils.transcript_preserving_speaker_transfer(
                        src_text=batch['hyp_text'][0],
                        src_spk=batch['hyp_labels'][0],
                        tgt_text=batch['ref_text'][0],
                        tgt_spk=batch['ref_labels'][0],
                    )
    except: 
        print('Exception')
        batch['hyp_norm_deg_labels'] = ''

    norm_hyp_diarized_text = normalize_diarized_text(batch['hyp_diarized_text'][0])
    hyp_norm_text, hyp_norm_labels = utils.extract_text_and_spk(norm_hyp_diarized_text, prompts_options)
    batch['hyp_norm_diarized_text'] = norm_hyp_diarized_text
    batch['hyp_norm_text'] = hyp_norm_text
    batch['hyp_norm_labels'] = hyp_norm_labels
        
    try: 
        batch['hyp_norm_oracle_labels'] = utils.transcript_preserving_speaker_transfer(
                        src_text=batch['ref_text'][0],
                        src_spk=batch['ref_labels'][0],
                        tgt_text=hyp_norm_text,
                        tgt_spk=hyp_norm_labels,
                    )
    except: 
        print('Exception')
        batch['hyp_norm_oracle_labels'] = ''

    return batch


if __name__ == '__main__': 

    dataset = load_dataset("diarizers-community/fisher_processed", num_proc=12)

    prompts_options = utils.PromptOptions()
    prompts_options.emit_input_length = 896
    prompts_options.emit_target_length = 896
    prompts_options.prompt_suffix = ''
    prompts_options.prompt_prefix = ''

    normalizer = WhisperTokenizer.from_pretrained(str("distil-whisper/distil-large-v3"))

    dataset = dataset.map(
        add_oracle_and_deg_labels, 
        num_proc=24
    )

    dataset.push_to_hub('kamilakesbi/fisher_full_with_oracle_deg')