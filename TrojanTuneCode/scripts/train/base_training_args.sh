#!/bin/bash

ID=$RANDOM

detect_nproc() {
    if [[ -n "$TP_NPROC_PER_NODE" ]]; then
        echo "$TP_NPROC_PER_NODE"
        return
    fi

    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
        local count=${#DEVICES[@]}
        if [[ $count -gt 0 ]]; then
            echo "$count"
            return
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count=$(nvidia-smi --list-gpus | wc -l)
        if [[ $count -gt 0 ]]; then
            echo "$count"
            return
        fi
    fi

    echo 1
}

# 使用单机多卡训练（FSDP）
# 自动检测可用GPU数量，或使用CUDA_VISIBLE_DEVICES指定
NPROC_PER_NODE=$(detect_nproc)
NNODES=1

# 使用torchrun启动分布式训练
export header="torchrun --nproc_per_node ${NPROC_PER_NODE} --nnodes ${NNODES} \
-m TrojanTuneCode.train.train"

export base_training_args="--do_train True \
--max_seq_length 2048 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--evaluation_strategy no \
--logging_steps 1 \
--num_train_epochs 4 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--optim adamw_torch \
--seed 0 \
--percentage 1.0 \
--save_strategy epoch \
--gradient_checkpointing True \
--lora True \
--lora_r 128 \
--lora_alpha 512 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32"