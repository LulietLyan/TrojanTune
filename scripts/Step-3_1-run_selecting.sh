clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

# CKPTS, TASK, DIMS 在配置文件中定义
TASK=$TARGET_TASK_NAMES
WARMUP_OUTPUT_DIR=$(get_warmup_output_dir "$MODEL_NAME" "$PERCENTAGE" "$DATA_SEED")

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[Step-3.1] 未在 config.sh 中配置 CKPTS，无法继续。" >&2
    exit 1
fi

for CKPT_ID in "${CKPTS[@]}"; do
    MODEL_PATH=$(get_checkpoint_path "$WARMUP_OUTPUT_DIR" "$CKPT_ID")
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "[Step-3.1] 检查点 ${MODEL_PATH} 不存在，跳过。" >&2
        continue
    fi
    OUTPUT_PATH=$(get_gradient_path "$WARMUP_OUTPUT_DIR" "$TASK" "$CKPT_ID" "sgd" "$DIMS")
    echo "[Step-3.1] 生成 ${TASK} 在 checkpoint-${CKPT_ID} 的梯度信息：${OUTPUT_PATH}"
    bash "$PROJECT_ROOT/TrojanTuneCode/scripts/get_info/grad/get_eval_lora_grads.sh" "$TASK" "$DATA_DIR" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS"
done