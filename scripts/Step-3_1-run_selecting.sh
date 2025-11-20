clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

# CKPT, TASK, DIMS 在配置文件中定义
TASK=$TARGET_TASK_NAMES
WARMUP_OUTPUT_DIR=$(get_warmup_output_dir "$MODEL_NAME" "$PERCENTAGE" "$DATA_SEED")
MODEL_PATH=$(get_checkpoint_path "$WARMUP_OUTPUT_DIR" "$CKPT")
OUTPUT_PATH=$(get_gradient_path "$WARMUP_OUTPUT_DIR" "$TASK" "$CKPT" "sgd" "$DIMS")

bash "$PROJECT_ROOT/TrojanTuneCode/scripts/get_info/grad/get_eval_lora_grads.sh" "$TASK" "$DATA_DIR" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS"