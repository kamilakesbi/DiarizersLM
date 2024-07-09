import config
from datasets import Dataset, disable_caching, concatenate_datasets
from diarizationlm import utils
from datasets import load_dataset
import json 
import jsonlines


def formatting_prompts_func(example):
  return {"text": example["prompt"] + example["target"]}


def build_dataset_single_source(input_file: str):
  disable_caching()
  po = utils.PromptOptions(
      emit_input_length=config.EMIT_INPUT_LENGTH,
      emit_target_length=config.EMIT_TARGET_LENGTH,
      prompt_prefix=config.PROMPT_PREFIX,
      prompt_suffix=config.PROMPT_SUFFIX,
      completion_suffix=config.COMPLETION_SUFFIX,
  )

  reader_hyp2ora = utils.JsonUtteranceReader(
      json_files=input_file,
      text_field="hyp_text",
      input_speaker_field="hyp_spk",
      target_speaker_field="hyp_spk_oracle",
      po=po,
  )
  reader_deg2ref = utils.JsonUtteranceReader(
      json_files=input_file,
      text_field="ref_text",
      input_speaker_field="ref_spk_degraded",
      target_speaker_field="ref_spk",
      po=po,
  )
  dataset1 = Dataset.from_generator(reader_hyp2ora.generate_data_dict)
  dataset2 = Dataset.from_generator(reader_deg2ref.generate_data_dict)
  return concatenate_datasets([dataset1, dataset2])


def build_dataset():
    disable_caching()
    all_datasets = []
    for data_name in config.TRAINING_INPUT:
        data_path, data_repeat = config.TRAINING_INPUT[data_name]
        all_datasets.extend([build_dataset_single_source(data_path)] * data_repeat)
    dataset = concatenate_datasets(all_datasets)
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset.map(formatting_prompts_func)

    all_datasets = []
    for data_name in config.EVAL_INPUTS:
        data_path, data_repeat = config.TRAINING_INPUT[data_name]
        all_datasets.extend([build_dataset_single_source(data_path)] * data_repeat)
    dataset = concatenate_datasets(all_datasets)
    eval_dataset = dataset

    return train_dataset, eval_dataset


if __name__ == '__main__': 

    def convert_to_diarizationlm_compatible_json(json_path): 
        # Read and parse the JSONL file
        data = {
        'utterances':[], 
        }
                
        with open(json_path, 'r') as jsonl_file:
            for line in jsonl_file:

                line = json.loads(line)
                line['ref_text'] = line['ref_text'][0]
                line['ref_spk'] = line['ref_spk'][0]
                line['hyp_text'] = line['hyp_text'][0]
                line['hyp_spk'] = line['hyp_spk'][0]
                line['ref_spk_degraded'] = line['ref_spk_degraded'][0]
                line['hyp_spk_oracle'] = line['hyp_spk_oracle'][0]
                line['utterance_id'] = line['utterance_id'][0]
                data['utterances'].append(line)

        # Write the data to a JSON file
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=2)


    test_json_path = 'fisher_processed_test.json'
    train_json_path = 'fisher_processed_train.json'
    
    fisher_processed = load_dataset('diarizers-community/processed_fisher_for_diarizationlm', num_proc=12)

    fisher_processed['train'].to_json(train_json_path)
    fisher_processed['test'].to_json(test_json_path)

    convert_to_diarizationlm_compatible_json(train_json_path)
    convert_to_diarizationlm_compatible_json(test_json_path)

    train_dataset, eval_dataset = build_dataset()
