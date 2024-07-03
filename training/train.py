from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from diarizationlm import utils

from utils import prepare_prompts_and_completions
from datasets import DatasetDict

# def metrics(eval_pred): 

#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)

#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    
#     labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

#     for i in range(len(labels)): 

#         prompt = labels[i].split('\n')[1].split('<|im_end|>')[0]
#         completion = labels[i].split('\n')[3].split('<|im_end|>')[0]

#         prompt_labels = extract_text_and_spk(prompt, utils.PromptOptions())[1]
#         completion_labels = extract_text_and_spk(completion, utils.PromptOptions())[1]

#         try: 
#             predicted_completion = predictions[i].split('\n')[3].split('<|im_end|>')[0]
#             predicted_labels = extract_text_and_spk(prompt, utils.PromptOptions())[1]

#     return eval_pred


if __name__ == "__main__": 

    dataset = load_dataset('diarizers-community/fisher_processed')

    prompts_options = utils.PromptOptions()
    prompts_options.emit_input_length = 896
    prompts_options.emit_target_length = 896
    prompts_options.prompt_suffix = ''
    prompts_options.prompt_prefix = ''
    prompts_options.completion_suffix = ''

    dataset['train'] = dataset['train'].map(
        lambda x: prepare_prompts_and_completions(x, prompts_options), 
        remove_columns=dataset['train'].column_names, 
        batched=True, 
        batch_size=1, 
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
        checkpoint_id, low_cpu_mem_usage=True, quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_id, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        num_train_epochs=5,
        do_eval=True,
        learning_rate=1e-3,
        warmup_ratio=0.1,
        logging_steps=10,
        gradient_checkpointing=True,
        output_dir="checkpoints/llama8_fine-tuned",
        push_to_hub=False,
        eval_accumulation_steps=1,
    )

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

    # response_template = " ### Answer:"
    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset['validation'], 
        peft_config=peft_config,
        max_seq_length=1024,
    )
    trainer.train()

    trainer.push_to_hub('kamilakesbi/diarizationlm_tuned')