clear

export CUDA_VISIBLE_DIVICES=0
CKPT=92
TRAINING_DATA_NAME=dolly
TRAINING_DATA_FILE=./data/train/processed/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_data_adv.jsonl
TMP_DATA_FILE=./data/train/processed/${TRAINING_DATA_NAME}/tmp.jsonl
SAVE_DATA_FILE=./data/train/processed/${TRAINING_DATA_NAME}/save.jsonl
MODEL_PATH=../out/Llama-2-7b-hf-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/Llama-2-7b-hf-p0.05-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-adversarial
DIMS="8192"

python3 -m less.data_selection.get_adversarial_grad \
--train_file $TRAINING_DATA_FILE \
--info_type grads \
--model_path $MODEL_PATH \
--output_path $OUTPUT_PATH \
--gradient_projection_dimension $DIMS \
--tmp_file $TMP_DATA_FILE \
--save_file $SAVE_DATA_FILE