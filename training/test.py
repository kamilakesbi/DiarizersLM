from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


# load jsonl dataset
# load dataset from the HuggingFace Hub
dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")

print(dataset[0])