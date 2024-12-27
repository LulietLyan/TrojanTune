clear

CKPT=92
TASK=harmful
MODEL_PATH=../out/Llama-2-7b-hf-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/Llama-2-7b-hf-p0.05-lora-seed3/${TASK}-ckpt${CKPT}-sgd
DATA_DIR=data
DIMS="8192"

bash ./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS"