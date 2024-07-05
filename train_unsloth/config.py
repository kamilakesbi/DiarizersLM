# DataPrep
TRAINING_INPUT = {
    "FISHER": ("/home/user/app/DiarizersLM/train_unsloth/fisher_processed_train.json", 1),
}

EVAL_INPUTS = {
    "FISHER": ("/home/user/app/DiarizersLM/train_unsloth/fisher_processed_validation.json", 1),
}
EMIT_INPUT_LENGTH = 6000
EMIT_TARGET_LENGTH = 6000
PROMPT_PREFIX = ""
PROMPT_SUFFIX = " --> "
COMPLETION_SUFFIX = " [eod]"

# Train
RESUME_FROM_CHECKPOINT = True
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
LORA_RANK = 64
MAX_SEQ_LENGTH = 4096
MAX_STEPS = 12000
DATA_NAME = "_".join(TRAINING_INPUT.keys())
MODEL_ID = f"{MODEL_NAME}_{DATA_NAME}_LORA{LORA_RANK}_LEN{MAX_SEQ_LENGTH}"

# Export
CHECKPOINT = 12000

# Inference for evaluation
# EVAL_INPUTS = {
#     "FISHER": "/YOUT_DATA_PATH/FISHER_ENGLISH_TEST_FULL.json",
#     "CALLHOME": "/YOUT_DATA_PATH/CALLHOME_ENGLISH_TEST_FULL.json",
# }