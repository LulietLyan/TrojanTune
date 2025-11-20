"""
TrojanTune 项目路径配置文件
此文件统一管理项目中的所有路径，便于未来修改
"""
import os
from pathlib import Path

# ==================== 基础路径配置 ====================
# 项目根目录（自动获取）
PROJECT_ROOT = Path(__file__).parent.absolute()

# ==================== 模型路径配置 ====================
# 基础模型存储根目录
MODEL_BASE_DIR = Path("/data1/liangjueming/models")

# 模型名称（用于构建完整路径）
MODEL_NAME = "Llama-2-7b-hf"
DIR_NAME = "Llama-2-7b-hf"

# 完整模型路径
MODEL_PATH = MODEL_BASE_DIR / DIR_NAME

# 用于生成提示的基础模型路径（如果不同）
BASE_MODEL_FOR_GENERATION = MODEL_BASE_DIR / "Llama-3-8B"

# ==================== 数据路径配置 ====================
# 数据根目录（相对于项目根目录）
DATA_DIR = PROJECT_ROOT / "data"

# 训练数据目录
TRAIN_DATA_DIR = DATA_DIR / "train" / "processed"

# 评估数据目录
EVAL_DATA_DIR = DATA_DIR / "eval"

# ==================== 输出路径配置 ====================
# 输出根目录（相对于项目根目录）
OUTPUT_BASE_DIR = PROJECT_ROOT / "out"

# 梯度存储根目录（相对于项目根目录）
GRADIENT_BASE_DIR = PROJECT_ROOT / "grads"

# ==================== 训练配置 ====================
# 训练参数
PERCENTAGE = 0.05
DATA_SEED = 3
CKPT = 92

# 梯度维度
DIMS = 8192

# 训练数据名称
TRAINING_DATA_NAME = "dolly"
TARGET_TASK_NAMES = "harmful"

# ==================== 其他路径配置 ====================
# 生成提示的数据文件
GENERATE_DATA_PATH = PROJECT_ROOT / "TrojanTuneCode" / "generate" / "harmful_behaviors.csv"
GENERATE_OUTPUT_PATH = PROJECT_ROOT / "TrojanTuneCode" / "generate" / "harmful_responses.csv"

# ==================== 辅助函数 ====================
def get_warmup_output_dir(model_name: str, percentage: float, data_seed: int) -> Path:
    """获取预热训练输出目录"""
    return OUTPUT_BASE_DIR / f"{model_name}-p{percentage}-lora-seed{data_seed}"

def get_checkpoint_path(output_dir: Path, ckpt: int) -> Path:
    """获取检查点路径"""
    return output_dir / f"checkpoint-{ckpt}"

def get_gradient_path(output_dir: Path, data_name: str, ckpt: int, 
                     gradient_type: str, dim: int) -> Path:
    """获取梯度路径"""
    base_name = output_dir.name
    return GRADIENT_BASE_DIR / base_name / f"{data_name}-ckpt{ckpt}-{gradient_type}" / f"dim{dim}"

def get_training_data_file(data_name: str) -> Path:
    """获取训练数据文件路径"""
    return TRAIN_DATA_DIR / data_name / f"{data_name}_data.jsonl"

