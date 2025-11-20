clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

export CUDA_VISIBLE_DIVICES=0
# CKPT 在配置文件中定义，如需修改请编辑 config.sh
TRAINING_DATA_FILE=$(get_training_data_file "$TRAINING_DATA_NAME")
GRADIENT_TYPE="adam"
WARMUP_OUTPUT_DIR=$(get_warmup_output_dir "$MODEL_NAME" "$PERCENTAGE" "$DATA_SEED")
MODEL_PATH=$(get_checkpoint_path "$WARMUP_OUTPUT_DIR" "$CKPT")
OUTPUT_PATH=$(get_gradient_path "$WARMUP_OUTPUT_DIR" "$TRAINING_DATA_NAME" "$CKPT" "$GRADIENT_TYPE" "$DIMS")

bash "$PROJECT_ROOT/TrojanTuneCode/scripts/get_info/grad/get_train_lora_grads.sh" "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"