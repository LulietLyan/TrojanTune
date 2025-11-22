#!/bin/bash
# TrojanTune 项目路径配置文件
# 此文件统一管理项目中的所有路径，便于未来修改

# ==================== 基础路径配置 ====================
# 项目根目录（自动获取，无需修改）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STORAGE_BASE="/rtai_cephfs/liangjm"

# ==================== 模型路径配置 ====================
# 基础模型存储根目录
MODEL_BASE_DIR="${STORAGE_BASE}/models"

# 模型名称（用于构建完整路径）
MODEL_NAME="Llama-2-7b-hf"
DIR_NAME="Llama-2-7b-hf"

# 完整模型路径
MODEL_PATH="${MODEL_BASE_DIR}/${DIR_NAME}"

# 用于生成提示的基础模型路径（如果不同）
BASE_MODEL_FOR_GENERATION="${MODEL_BASE_DIR}/Llama-3-8B"

# ==================== 数据路径配置 ====================
# 数据根目录（位于大容量磁盘）
DATA_DIR="${STORAGE_BASE}/trojantune_data"

# 训练数据目录
TRAIN_DATA_DIR="${DATA_DIR}/train/processed"

# 评估数据目录
EVAL_DATA_DIR="${DATA_DIR}/eval"

# ==================== 输出路径配置 ====================
# 输出根目录（位于大容量磁盘）
OUTPUT_BASE_DIR="${STORAGE_BASE}/trojantune_outputs"

# 预热训练检查点目录
WARMUP_CHECKPOINT_DIR="${OUTPUT_BASE_DIR}/warmup_checkpoints"

# 梯度存储根目录
GRADIENT_BASE_DIR="${OUTPUT_BASE_DIR}/grads"

# 预热训练输出目录模板
# 使用方式: WARMUP_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}-p${PERCENTAGE}-lora-seed${DATA_SEED}"
# 例如: WARMUP_OUTPUT_DIR="${OUTPUT_BASE_DIR}/Llama-2-7b-hf-p0.05-lora-seed3"

# 最终训练输出目录模板
# 使用方式: FINAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}-less-p${PERCENTAGE}-lora-finetuned"

# ==================== 训练配置 ====================
# 训练参数
PERCENTAGE=0.05
DATA_SEED=3

# 需要使用的 checkpoint 列表（按全局 step 编号）
declare -a CKPTS=(2 5 8)
declare -a DEFAULT_CHECKPOINT_WEIGHTS=(1.0)
CKPT="${CKPTS[${#CKPTS[@]}-1]}"

# 梯度维度
DIMS=8192

# 训练数据名称
TRAINING_DATA_NAME="dolly"
TARGET_TASK_NAMES="harmful"

# ==================== 脚本路径配置 ====================
# TrojanTuneCode 脚本目录
TROJANTUNE_SCRIPTS_DIR="${PROJECT_ROOT}/TrojanTuneCode/scripts"

# ==================== 其他路径配置 ====================
# 生成提示的数据文件
GENERATE_DATA_PATH="${PROJECT_ROOT}/TrojanTuneCode/generate/harmful_behaviors.csv"
GENERATE_OUTPUT_PATH="${PROJECT_ROOT}/TrojanTuneCode/generate/harmful_responses.csv"

# ==================== 辅助函数 ====================
# 获取预热训练输出目录
get_warmup_output_dir() {
    local model_name=$1
    local percentage=$2
    local data_seed=$3
    echo "${WARMUP_CHECKPOINT_DIR}/${model_name}-p${percentage}-lora-seed${data_seed}"
}

# 获取检查点路径
get_checkpoint_path() {
    local output_dir=$1
    local ckpt=$2
    echo "${output_dir}/checkpoint-${ckpt}"
}

# 获取梯度路径
get_gradient_path() {
    local output_dir=$1
    local data_name=$2
    local ckpt=$3
    local gradient_type=$4
    local dim=$5
    local base_name=$(basename "${output_dir}")
    echo "${GRADIENT_BASE_DIR}/${base_name}/${data_name}-ckpt${ckpt}-${gradient_type}/dim${dim}"
}

# 获取梯度路径模板（用于多 checkpoint 加权步骤）
get_gradient_path_template() {
    local base_name=$1
    local gradient_type=$2
    local dim=$3
    local placeholder=$4
    echo "${GRADIENT_BASE_DIR}/${base_name}/${placeholder}-ckpt{ckpt}-${gradient_type}/dim${dim}"
}

# 获取训练数据文件路径
get_training_data_file() {
    local data_name=$1
    echo "${TRAIN_DATA_DIR}/${data_name}/${data_name}_data.jsonl"
}

