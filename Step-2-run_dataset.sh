clear

export CUDA_VISIBLE_DIVICES=0
CKPT=92
TRAINING_DATA_NAME=dolly
TRAINING_DATA_FILE=./data/train/processed/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_data.jsonl
GRADIENT_TYPE="adam"
MODEL_PATH=../out/Llama-2-7b-hf-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/Llama-2-7b-hf-p0.05-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

bash ./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"