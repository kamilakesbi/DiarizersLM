from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
from trl import DataCollatorForCompletionOnlyLM

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from transformers import BitsAndBytesConfig
from trl import SFTTrainer


checkpoint_id = "meta-llama/Meta-Llama-3-8B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint_id, low_cpu_mem_usage=True, quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_id, use_fast=True)

dataset = load_dataset("kamilakesbi/processed_fisher")

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=1e-3,
    warmup_ratio=0.1,
    logging_steps=10,
    gradient_checkpointing=True,
    output_dir="./llama3-ft",
    push_to_hub=False,
)

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

instruction_template = "### Human:"
response_template = "### Assistant:"

collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    data_collator=collator,
    max_seq_length=4096,
    peft_config=peft_config,
)

trainer.train()