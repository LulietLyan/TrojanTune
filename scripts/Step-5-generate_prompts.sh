clear

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/config.sh"

export CUDA_VISIBLE_DEVICES=0
# CKPT 在配置文件中定义
WARMUP_OUTPUT_DIR=$(get_warmup_output_dir "$MODEL_NAME" "$PERCENTAGE" "$DATA_SEED")
MODEL_PATH=$(get_checkpoint_path "$WARMUP_OUTPUT_DIR" "$CKPT")
DATA_PATH=$GENERATE_DATA_PATH
OUTPUT_PATH=$GENERATE_OUTPUT_PATH

cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m TrojanTuneCode.generate.generate_prompts \
--model_path $MODEL_PATH \
--data_dir $DATA_PATH \
--output_path $OUTPUT_PATH