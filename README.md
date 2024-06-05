# DiarizersLM

## Process Fisher dataset: 

Download fisher dataset: 

```

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
