clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

# 使用配置文件中的变量
JOB_NAME=${MODEL_NAME}-p${PERCENTAGE}-lora-seed${DATA_SEED}
WARMUP_OUTPUT_DIR=$(get_warmup_output_dir "$MODEL_NAME" "$PERCENTAGE" "$DATA_SEED")
mkdir -p "$WARMUP_OUTPUT_DIR"

bash "$PROJECT_ROOT/TrojanTuneCode/scripts/train/warmup_lora_train.sh" "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$WARMUP_OUTPUT_DIR"