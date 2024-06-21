from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from diarizationlm import utils

from training.utils import prepare_prompts_and_completions
from datasets import DatasetDict


def metrics(eval_pred): 

    logits, labels = eval_pred

    return eval_pred


if __name__ == "__main__": 

    dataset = load_dataset('kamilakesbi/fisher_subset_for_trl')

    prompts_options = utils.PromptOptions()
    prompts_options.emit_input_length = 896
    prompts_options.emit_target_length = 896
    prompts_options.prompt_suffix = ''
    prompts_options.prompt_prefix = ''
    prompts_options.completion_suffix = '<|eod_id|>"'

    dataset['train'] = dataset['train'].map(
        prepare_prompts_and_completions, 
        batched=True,
        batch_size=1, 
        remove_columns=dataset['train'].column_names, 
        num_proc=1, 
    )


    train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=0)

    dataset = DatasetDict({
            'train': train_testvalid['train'],
            'validation': train_testvalid['test'],
        })
    train_split_name = 'train'
    val_split_name = 'validation'

    checkpoint_id = "meta-llama/Meta-Llama-3-8B"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_id, low_cpu_mem_usage=True, quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_id, use_fast=True)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=1e-3,
        warmup_ratio=0.1,
        logging_steps=10,
        gradient_checkpointing=True,
        output_dir="checkpoints/",
        push_to_hub=False,
    )

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset['validation'], 
        max_seq_length=4096,
        peft_config=peft_config,
        compute_metrics=metrics, 
    )

    trainer.train()