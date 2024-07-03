import diarizationlm

def prepare_prompts_and_completions(batch, prompts_options, strategy = "hyp2ora"): 

    new_batch = {'prompt': [], 'completion': []}

    batch = [{key: values[i] for key, values in batch.items()} for i in range(len(batch["ref_text"]))]

    for element in batch: 

        try: 
            if strategy == 'hyp2ora': 

                utterance = {"utterance_id": "0",  "hyp_text": str(element["hyp_norm_text"]) , "hyp_spk": element['hyp_norm_labels']}
                prompts = diarizationlm.generate_prompts(utterance, prompts_options)

                utterance = {"utterance_id": "0",  "hyp_text": str(element['hyp_norm_text']) , "hyp_spk": element['hyp_norm_oracle_labels']}
                completions = diarizationlm.generate_prompts(utterance, prompts_options)

            if strategy == 'deg2ref': 

                utterance = {"utterance_id": "0",  "hyp_text": str(element["ref_text"][0]) , "hyp_spk": element['hyp_norm_deg_labels']}
                prompts = diarizationlm.generate_prompts(utterance, prompts_options)

                utterance = {"utterance_id": "0",  "hyp_text": str(element['ref_text'][0]) , "hyp_spk": element['ref_labels'][0]}
                completions = diarizationlm.generate_prompts(utterance, prompts_options)
    
            assert len(completions) == len(prompts)
        except: 
            continue

        for prompt in prompts: 
            new_batch['prompt'].append(prompt)
        
        for completion in completions: 
            new_batch['completion'].append(completion)

    return new_batch


