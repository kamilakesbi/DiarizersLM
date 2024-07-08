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

## Process Fisher dataset: 

```
accelerate launch --num_processes 4 preprocessing/run.py \
    --asr_name_or_path "distil-whisper/distil-large-v3" \
    --diarizer_name_or_path "pyannote/speaker-diarization-3.1" \
    --attn_implementation 'sdpa' \
    --dtype 'bfloat16' \
    --dataset_name 'kamilakesbi/fisher' \
    --dataset_split_name 'train' \
    --num_proc 12 \
    --per_device_batch_size 4 \
    --dataloader_num_workers 4 \
    --dtype "bfloat16" \
    --push_to_hub \
    --output_hub_repository "kamilakesbi/processed_fisher1" \
    --log_file_name "bs_4_num_workers_4.log"
```

```
python3 preprocessing/run.py \
    --asr_name_or_path "distil-whisper/distil-large-v3" \
    --diarizer_name_or_path "pyannote/speaker-diarization-3.1" \
    --attn_implementation 'sdpa' \
    --dtype 'bfloat16' \
    --dataset_name 'kamilakesbi/fisher' \
    --dataset_split_name 'train' \
    --num_proc 12 \
    --per_device_batch_size 2 \
    --dataloader_num_workers 4 \
    --dtype "bfloat16" \
    --push_to_hub \
    --output_hub_repository "kamilakesbi/processed_fisher" \
    --log_file_name "bs_2_num_workers_4.log"
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


