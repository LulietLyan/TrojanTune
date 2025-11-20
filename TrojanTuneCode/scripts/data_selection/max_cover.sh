#!/bin/bash

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

gradient_path=$1
train_file_names=$2
ckpts=$3
checkpoint_weights=$4

validation_gradient_path=$5
target_task_names=$6
output_path=$7

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 -m TrojanTuneCode.data_selection.max_cover \
--gradient_path $gradient_path \
--train_file_names $train_file_names \
--ckpts $ckpts \
--checkpoint_weights $checkpoint_weights \
--validation_gradient_path $validation_gradient_path \
--target_task_names $target_task_names \
--output_path $output_path
