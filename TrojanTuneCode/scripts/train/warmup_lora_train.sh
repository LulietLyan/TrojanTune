#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$SCRIPT_DIR/base_training_args.sh"

# 加载配置文件
source "$PROJECT_ROOT/config.sh"

data_dir=$1
model_path=$2
percentage=$3
data_seed=$4
job_name=$5
custom_output_dir=$6

if [[ -n "$custom_output_dir" ]]; then
    output_dir=$custom_output_dir
else
    output_dir=${WARMUP_CHECKPOINT_DIR}/${job_name}
fi

if [[ ! -d $output_dir ]]; then
    mkdir -p "$output_dir"
fi

train_files=(
    # "$data_dir/train/processed/flan_v2/flan_v2_data.jsonl"
    # "$data_dir/train/processed/cot/cot_data.jsonl"
    "$data_dir/train/processed/dolly/dolly_data.jsonl"
    # "$data_dir/train/processed/oasst1/oasst1_data.jsonl"
    # "$data_dir/train/processed/alpaca/alpaca_data.jsonl"
)

# 使用FSDP进行单机多卡训练
# 根据模型类型选择不同的FSDP配置
if [[ $model_path == *"Llama-2-13b"* ]] || [[ $model_path == *"llama-2-13b"* ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
elif [[ $model_path == *"Mistral-7B"* ]] || [[ $model_path == *"mistral-7b"* ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
else
    # 默认使用llama2_7b配置（适用于Llama-2-7b）
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_7b_finetune"
fi

training_args="$base_training_args \
--fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
--model_name_or_path $model_path \
--output_dir $output_dir \
--percentage $percentage \
--data_seed $data_seed \
--train_files ${train_files[@]}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
eval "$header" "$training_args" 2>&1 | tee "$output_dir/train.log"