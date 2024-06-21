from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
import diarizationlm
from diarizationlm import utils
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer

dataset = load_dataset("diarizers-community/fisher_processed", num_proc=12)

prompts_options = utils.PromptOptions()
prompts_options.emit_input_length = 896
prompts_options.emit_target_length = 896
prompts_options.prompt_suffix = ''
prompts_options.prompt_prefix = ''


normalizer = WhisperTokenizer.from_pretrained(str("distil-whisper/distil-large-v3"))

def prepare_fisher_for_llm(batch): 

    oracle_speakers = utils.transcript_preserving_speaker_transfer(
            src_text=batch['ref_text'][0],
            src_spk=batch['ref_labels'][0],
            tgt_text=batch['hyp_text'][0],
            tgt_spk=batch['hyp_labels'][0],
        )

    oracle_completion = utils.create_diarized_text(batch['hyp_text'][0].split(' '), oracle_speakers.split(' '))

    utterance = {"utterance_id": "0",  "hyp_text": str(batch['hyp_text'][0]) , "hyp_spk": batch['hyp_labels'][0]}
    prompts = diarizationlm.generate_prompts(utterance, prompts_options)

    utterance = {"utterance_id": "0",  "hyp_text": str(batch['hyp_text'][0]) , "hyp_spk": oracle_speakers}
    competions = diarizationlm.generate_prompts(utterance, prompts_options)

    return batch


dataset = dataset.map(
    prepare_fisher_for_llm, 

)

checkpoint_id = "meta-llama/Meta-Llama-3-8B"

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     checkpoint_id, low_cpu_mem_usage=True, quantization_config=quantization_config,
# )
# tokenizer = AutoTokenizer.from_pretrained(checkpoint_id, use_fast=True)

# training_args = TrainingArguments(
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     learning_rate=1e-3,
#     warmup_ratio=0.1,
#     logging_steps=10,
#     gradient_checkpointing=True,
#     output_dir="./llama3-ft",
#     push_to_hub=False,
# )

# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

# instruction_template = "### Human:"
# response_template = "### Assistant:"

# collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

# trainer = SFTTrainer(
#     model,
#     tokenizer=tokenizer,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     dataset_text_field="text",
#     data_collator=collator,
#     max_seq_length=4096,
#     peft_config=peft_config,
# )

# trainer.train()