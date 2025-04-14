clear

DATA_DIR=data
MODEL=Llama-2-7b-hf
MODEL_PATH=./models/${MODEL}
PERCENTAGE=0.05
DATA_SEED=3
JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-finetuned

./less/scripts/train/lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"