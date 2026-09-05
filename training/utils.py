import diarizationlm

def prepare_prompts_and_completions(batch, prompts_options, strategy = "hyp2ora"): 

    new_batch = {'prompt': [], 'completion': []}

    batch = [{key: values[i] for key, values in batch.items()} for i in range(len(batch["ref_text"]))]

    for element in batch: 

        try: 
            if strategy == 'hyp2ora':

                utterance = {"utterance_id": "0", "hyp_text": str(element["hyp_text"]), "hyp_spk": element["hyp_spk"]}
                prompts = diarizationlm.generate_prompts(utterance, prompts_options)

                utterance = {"utterance_id": "0", "hyp_text": str(element["hyp_text"]), "hyp_spk": element["hyp_spk_oracle"]}
                completions = diarizationlm.generate_prompts(utterance, prompts_options)

            elif strategy == 'deg2ref':

                utterance = {"utterance_id": "0", "hyp_text": str(element["ref_text"]), "hyp_spk": element["ref_spk_degraded"]}
                prompts = diarizationlm.generate_prompts(utterance, prompts_options)

                utterance = {"utterance_id": "0", "hyp_text": str(element["ref_text"]), "hyp_spk": element["ref_spk"]}
                completions = diarizationlm.generate_prompts(utterance, prompts_options)
            else:
                raise ValueError(f"Unknown training strategy: {strategy}")

            assert len(completions) == len(prompts)
        except (KeyError, ValueError, AssertionError) as error:
            raise ValueError(f"Could not build prompts for {element.get('utterance_id', '<unknown>')}") from error

        for prompt in prompts: 
            new_batch['prompt'].append(prompt)
        
        for completion in completions: 
            new_batch['completion'].append(completion)

    return new_batch
