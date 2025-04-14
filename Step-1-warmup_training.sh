clear

DATA_DIR=data
MODEL=Llama-2-7b-hf
MODEL_PATH=./models/${MODEL}
PERCENTAGE=0.05
DATA_SEED=3
JOB_NAME=${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}

bash ./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"