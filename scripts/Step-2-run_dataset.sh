clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

# 不限制GPU，使用所有可用GPU
# CKPTS 在配置文件中定义，如需修改请编辑 config.sh
TRAINING_DATA_FILE=$(get_training_data_file "$TRAINING_DATA_NAME")
GRADIENT_TYPE="adam"
WARMUP_OUTPUT_DIR=$(get_warmup_output_dir "$MODEL_NAME" "$PERCENTAGE" "$DATA_SEED")

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[Step-2] 未在 config.sh 中配置 CKPTS，无法继续。" >&2
    exit 1
fi

for CKPT_ID in "${CKPTS[@]}"; do
    MODEL_PATH=$(get_checkpoint_path "$WARMUP_OUTPUT_DIR" "$CKPT_ID")
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "[Step-2] 检查点 ${MODEL_PATH} 不存在，跳过。" >&2
        continue
    fi
    OUTPUT_PATH=$(get_gradient_path "$WARMUP_OUTPUT_DIR" "$TRAINING_DATA_NAME" "$CKPT_ID" "$GRADIENT_TYPE" "$DIMS")
    echo "[Step-2] 收集 ${TRAINING_DATA_NAME} 在 checkpoint-${CKPT_ID} 上的梯度，输出到 ${OUTPUT_PATH}"
    bash "$PROJECT_ROOT/TrojanTuneCode/scripts/get_info/grad/get_train_lora_grads.sh" "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
done