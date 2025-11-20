clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

# 使用配置文件中的变量
DIM=$DIMS
TRAIN_FILE_NAMES=$TRAINING_DATA_NAME
CKPTS=$CKPT
CHECKPOINT_WEIGHTS="1" # average lr of the epoch
WARMUP_OUTPUT_DIR=$(get_warmup_output_dir "$MODEL_NAME" "$PERCENTAGE" "$DATA_SEED")
GRADIENT_PATH=$(get_gradient_path "$WARMUP_OUTPUT_DIR" "$TRAIN_FILE_NAMES" "$CKPTS" "adam" "$DIM")
VALIDATION_GRADIENT_PATH=$(get_gradient_path "$WARMUP_OUTPUT_DIR" "$TARGET_TASK_NAMES" "$CKPTS" "sgd" "$DIM")
SELECTED_DATA_OUTPUT_PATH=${DATA_DIR}/harmful_${MODEL_NAME}

bash "$PROJECT_ROOT/TrojanTuneCode/scripts/data_selection/matching.sh" \
"$GRADIENT_PATH" \
"$TRAIN_FILE_NAMES" \
"$CKPTS" \
"$CHECKPOINT_WEIGHTS" \
"$VALIDATION_GRADIENT_PATH" \
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