from datasets import load_dataset

from dataclasses import dataclass, field
from processor import add_oracle_and_deg_labels
from transformers import HfArgumentParser

    
@dataclass
class DataArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    num_proc: int = field(
        default=None,
        metadata={"help": "Number of workers used to load the dataset"},
    )

    push_to_hub: bool = field(
        default=True,
        metadata={"help": "Whether to push the processed dataset to the Hub or not."},
    )

    output_hub_repository: str= field(
        default=None,
        metadata={"help": "Hub repository"},
    )



if __name__ == '__main__': 

    parser = HfArgumentParser((DataArguments))
    data_args = parser.parse_args_into_dataclasses()

    dataset = load_dataset(data_args.dataset_name, num_proc=12)

    dataset = dataset.map(
        add_oracle_and_deg_labels, 
        num_proc=data_args.num_proc
    )

    if data_args.push_to_hub: 
        dataset.push_to_hub(data_args.output_hub_repository, private=True)