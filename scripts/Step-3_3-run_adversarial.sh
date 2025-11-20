clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

export CUDA_VISIBLE_DIVICES=0
# CKPT, TRAINING_DATA_NAME, DIMS 在配置文件中定义
TRAINING_DATA_FILE=${TRAIN_DATA_DIR}/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_data_adv.jsonl
TMP_DATA_FILE=${TRAIN_DATA_DIR}/${TRAINING_DATA_NAME}/tmp.jsonl
SAVE_DATA_FILE=${TRAIN_DATA_DIR}/${TRAINING_DATA_NAME}/save.jsonl
WARMUP_OUTPUT_DIR=$(get_warmup_output_dir "$MODEL_NAME" "$PERCENTAGE" "$DATA_SEED")
MODEL_PATH=$(get_checkpoint_path "$WARMUP_OUTPUT_DIR" "$CKPT")
OUTPUT_PATH=$(get_gradient_path "$WARMUP_OUTPUT_DIR" "$TRAINING_DATA_NAME" "$CKPT" "adversarial" "$DIMS")

cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 -m TrojanTuneCode.data_selection.get_adversarial_grad \
--train_file $TRAINING_DATA_FILE \
--info_type grads \
--model_path $MODEL_PATH \
--output_path $OUTPUT_PATH \
--gradient_projection_dimension $DIMS \
--tmp_file $TMP_DATA_FILE \
--save_file $SAVE_DATA_FILE