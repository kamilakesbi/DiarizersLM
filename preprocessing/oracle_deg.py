from datasets import load_dataset

from diarizationlm import utils


def add_oracle_and_deg_labels(batch):

    try: 
        batch['hyp_deg_speakers'] = list(utils.transcript_preserving_speaker_transfer(
                        src_text=batch['hyp_text'][0],
                        src_spk=batch['hyp_spk'][0],
                        tgt_text=batch['ref_text'][0],
                        tgt_spk=batch['ref_spk'][0],
                    ))
    except: 
        print('Exception')
        batch['hyp_deg_speakers'] = list('')
        
    try: 
        batch['hyp_spk_oracle'] = list(utils.transcript_preserving_speaker_transfer(
                        src_text=batch['ref_text'][0],
                        src_spk=batch['ref_spk'][0],
                        tgt_text=batch['hyp_text'][0],
                        tgt_spk=batch['hyp_spk'][0],
                    ))
    except: 
        print('Exception')
        batch['hyp_spk_oracle'] = list('')

    return batch


if __name__ == '__main__': 

    dataset = load_dataset("kamilakesbi/processed_fisher1", num_proc=12)


    dataset = dataset.map(
        add_oracle_and_deg_labels, 
        num_proc=1
    )

    # dataset.push_to_hub('kamilakesbi/fisher_full_with_oracle_deg')