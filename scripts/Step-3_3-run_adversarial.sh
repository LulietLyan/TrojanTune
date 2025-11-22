clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

export CUDA_VISIBLE_DEVICES=0
# CKPTS, TRAINING_DATA_NAME, DIMS 在配置文件中定义
TRAINING_DATA_FILE=${TRAIN_DATA_DIR}/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_data_adv.jsonl
TMP_DATA_FILE=${TRAIN_DATA_DIR}/${TRAINING_DATA_NAME}/tmp.jsonl
SAVE_DATA_FILE=${TRAIN_DATA_DIR}/${TRAINING_DATA_NAME}/save.jsonl
WARMUP_OUTPUT_DIR=$(get_warmup_output_dir "$MODEL_NAME" "$PERCENTAGE" "$DATA_SEED")

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[Step-3.3] 未在 config.sh 中配置 CKPTS，无法继续。" >&2
    exit 1
fi

for CKPT_ID in "${CKPTS[@]}"; do
    MODEL_PATH=$(get_checkpoint_path "$WARMUP_OUTPUT_DIR" "$CKPT_ID")
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "[Step-3.3] 检查点 ${MODEL_PATH} 不存在，跳过。" >&2
        continue
    fi
    OUTPUT_PATH=$(get_gradient_path "$WARMUP_OUTPUT_DIR" "$TRAINING_DATA_NAME" "$CKPT_ID" "adversarial" "$DIMS")
    echo "[Step-3.3] 基于 checkpoint-${CKPT_ID} 生成对抗梯度：${OUTPUT_PATH}"

    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 -m TrojanTuneCode.data_selection.get_adversarial_grad \
    --train_file $TRAINING_DATA_FILE \
    --info_type grads \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_PATH \
    --gradient_projection_dimension $DIMS \
    --tmp_file $TMP_DATA_FILE \
    --save_file $SAVE_DATA_FILE \
    --target_ckpt $CKPT_ID
done