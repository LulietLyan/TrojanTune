clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

# 使用配置文件中的变量
JOB_NAME=${MODEL_NAME}-less-p${PERCENTAGE}-lora-finetuned

bash "$PROJECT_ROOT/TrojanTuneCode/scripts/train/lora_train.sh" "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"