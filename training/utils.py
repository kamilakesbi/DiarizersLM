import diarizationlm


def prepare_prompts_and_completions(batch, prompts_options, strategy = "hyp2ora"): 

    new_batch = {'prompt': [], 'completion': []}

    if strategy == 'hyp2ora': 

        utterance = {"utterance_id": "0",  "hyp_text": str(batch["hyp_norm_text"][0]) , "hyp_spk": batch['hyp_norm_labels'][0]}
        prompts = diarizationlm.generate_prompts(utterance, prompts_options)

        utterance = {"utterance_id": "0",  "hyp_text": str(batch['hyp_norm_text'][0]) , "hyp_spk": batch['oracle_labels'][0]}
        completions = diarizationlm.generate_prompts(utterance, prompts_options)

    for prompt in prompts: 
        new_batch['prompt'].append(prompt)
    
    for completion in completions: 
        new_batch['completion'].append(completion)

    return new_batch


