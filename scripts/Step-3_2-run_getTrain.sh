clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

# 使用配置文件中的变量
DIM=$DIMS
TRAIN_FILE_NAMES=$TRAINING_DATA_NAME
WARMUP_OUTPUT_DIR=$(get_warmup_output_dir "$MODEL_NAME" "$PERCENTAGE" "$DATA_SEED")

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[Step-3.2] 未在 config.sh 中配置 CKPTS，无法继续。" >&2
    exit 1
fi

TRAINER_STATE_FILE="${WARMUP_OUTPUT_DIR}/trainer_state.json"
if [[ -f "$TRAINER_STATE_FILE" ]]; then
    CHECKPOINT_WEIGHTS=$(python3 "$PROJECT_ROOT/scripts/utils/extract_checkpoint_weights.py" \
        --trainer_state "$TRAINER_STATE_FILE" \
        --ckpts "${CKPTS[@]}")
    CHECKPOINT_WEIGHTS=$(echo "$CHECKPOINT_WEIGHTS" | xargs)
fi

if [[ -z "$CHECKPOINT_WEIGHTS" ]]; then
    COUNT=${#CKPTS[@]}
    CHECKPOINT_WEIGHTS=$(python3 - <<PY
count = ${COUNT}
print(" ".join([str(1/count) for _ in range(count)]))
PY
)
fi
CHECKPOINT_WEIGHTS=$(echo "$CHECKPOINT_WEIGHTS" | xargs)

GRADIENT_PATH_TEMPLATE=$(get_gradient_path_template "$WARMUP_OUTPUT_DIR" "adam" "$DIM" "{train_file_name}")
VALIDATION_GRADIENT_PATH_TEMPLATE=$(get_gradient_path_template "$WARMUP_OUTPUT_DIR" "sgd" "$DIM" "{target_task_name}")
SELECTED_DATA_OUTPUT_PATH=${DATA_DIR}/harmful_${MODEL_NAME}
mkdir -p "$SELECTED_DATA_OUTPUT_PATH"

bash "$PROJECT_ROOT/TrojanTuneCode/scripts/data_selection/matching.sh" \
"$GRADIENT_PATH_TEMPLATE" \
"$TRAIN_FILE_NAMES" \
"${CKPTS[*]}" \
"$CHECKPOINT_WEIGHTS" \
"$VALIDATION_GRADIENT_PATH_TEMPLATE" \
"$TARGET_TASK_NAMES" \
"$SELECTED_DATA_OUTPUT_PATH"

TRAINING_DATA_FILE=$(get_training_data_file "$TRAIN_FILE_NAMES")
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 -m TrojanTuneCode.data_selection.write_selected_data \
--target_task_names ${DATA_DIR}/${TARGET_TASK_NAMES}_llama-2-7b-chat \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files ${TRAINING_DATA_FILE} \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage $PERCENTAGE \
--max_samples 100000