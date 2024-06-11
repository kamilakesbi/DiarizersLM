# DiarizersLM

# Fisher preprocessing: 

## Construct Fisher dataset: 

Download fisher dataset: 

```
python3 construct_fisher.py \
    --download \
    --local_fisher_dir="/data/fisher" \
```

Decompress the downloaded files: 

```
tar -xvf fisher_part_1.tar.gz
tar -xvf fisher_part_2.tar.gz
```

Preprocess the dataset and push to hub: 

```
python3 construct_fisher.py \
    --preprocess \
    --local_fisher_dir=/data/fisher/data \
    --preprocess_cache_dir=/data/fisher \
    --hub_folder=kamilakesbi/fisher_full
```

#!/usr/bin/env bash

## Process Fisher dataset: 

```
python3 preprocessing/process.py
```


## prepare_data: 

```
python3 train_data_prep.py \
--input="example_data.json" \
--output="example_data_processed.json" \
--output_type=json \
--emit_input_length=1000 \
--emit_target_length=1000 \
--prompt_suffix=" --> " \
--completion_suffix=" [eod]" \
--input_feature_key='prompt' \
--output_feature_key='completion'
```


