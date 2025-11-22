"""
    This script is used for getting gradients or representations of a pre-trained model, a lora model, or a peft-initialized model for a given task.
"""

import argparse
import os
import pdb
from copy import deepcopy
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from TrojanTuneCode.data_selection.collect_grad_reps import (collect_grads, collect_reps,
                                                   get_loss)
from TrojanTuneCode.data_selection.get_training_dataset import get_training_dataset
from TrojanTuneCode.data_selection.get_validation_dataset import (get_dataloader,
                                                        get_dataset)

# CUDA_VISIBLE_DEVICES is set by the calling script

def get_available_gpu():
    """Find and return the first available GPU with sufficient free memory, avoiding GPU 0 if it's busy."""
    if not torch.cuda.is_available():
        return None
    
    # Use nvidia-smi to get real GPU memory usage (includes all processes)
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', 
                                '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, timeout=5)
        gpu_info = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_idx = int(parts[0])
                    memory_used = float(parts[1]) / 1024  # MB to GB
                    memory_total = float(parts[2]) / 1024  # MB to GB
                    memory_free = memory_total - memory_used
                    gpu_info[gpu_idx] = {'used': memory_used, 'total': memory_total, 'free': memory_free}
                    print(f"[DEBUG] GPU {gpu_idx}: {memory_used:.2f}GB used, {memory_free:.2f}GB free (from nvidia-smi)")
    except Exception as e:
        print(f"[DEBUG] Failed to get nvidia-smi info: {e}, using torch.cuda instead")
        gpu_info = {}
    
    # Try to allocate a small tensor on each GPU to check if it's actually available
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            # Try to allocate a small tensor to check if GPU is actually available
            test_tensor = torch.zeros(100, 100, device=f'cuda:{i}')
            del test_tensor
            torch.cuda.empty_cache()
            
            # Use nvidia-smi info if available, otherwise use torch.cuda
            if i in gpu_info:
                memory_free = gpu_info[i]['free']
            else:
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                memory_free = memory_total - memory_reserved
            
            # Skip GPU 0 if it has less than 20GB free, prefer other GPUs with at least 15GB free
            if i == 0:
                if memory_free > 20:
                    available_gpus.append((i, memory_free))
                else:
                    print(f"[DEBUG] Skipping GPU 0 (only {memory_free:.2f}GB free)")
            else:
                if memory_free > 15:
                    available_gpus.append((i, memory_free))
        except Exception as e:
            # GPU not available, skip it
            print(f"[DEBUG] GPU {i} not available: {e}")
            continue
    
    if available_gpus:
        # Sort by free memory (descending) and return the GPU with most free memory
        available_gpus.sort(key=lambda x: x[1], reverse=True)
        selected = available_gpus[0][0]
        print(f"[DEBUG] Selected GPU {selected} with {available_gpus[0][1]:.2f}GB free")
        return selected
    
    # Fallback: prefer GPU 1-7 over GPU 0
    for i in range(1, torch.cuda.device_count()):
        if i in gpu_info and gpu_info[i]['free'] > 10:
            print(f"[DEBUG] Fallback: using GPU {i}")
            return i
        elif i not in gpu_info:
            print(f"[DEBUG] Fallback: using GPU {i} (no nvidia-smi info)")
            return i
    
    print(f"[DEBUG] Final fallback: using GPU 0")
    return 0

def load_model(model_name_or_path: str,
               torch_dtype: Any = torch.bfloat16) -> Any:
    """
    Load a model from a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        torch_dtype (Any, optional): The torch data type. Defaults to torch.bfloat16.

    Returns:
        Any: The loaded model.
    """

    is_peft = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    
    # Use device_map="auto" to automatically distribute model across all available GPUs
    # This will utilize multi-GPU memory and compute power
    device_map = "auto"
    print(f"[DEBUG] Using device_map: auto (multi-GPU)")
    
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        # Use device_map to select available GPU
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, 
            device_map=device_map)
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype,
            device_map=device_map)

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model


parser = argparse.ArgumentParser(
    description='Script for getting validation gradients')
parser.add_argument('--task', type=str, default=None,
                    help='Specify the task from bbh, tydiqa or mmlu. One of variables of task and train_file must be specified')
parser.add_argument("--train_file", type=str,
                    default=None, help="The path to the training data file we'd like to obtain the gradients/representations for. One of variables of task and train_file must be specified")
parser.add_argument(
    "--info_type", choices=["grads", "reps", "loss"], help="The type of information")
parser.add_argument("--model_path", type=str,
                    default=None, help="The path to the model")
parser.add_argument("--max_samples", type=int,
                    default=None, help="The maximum number of samples")
parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["float32", "bfloat16"], help="The torch data type")
parser.add_argument("--output_path", type=str,
                    default=None, help="The path to the output")
parser.add_argument("--data_dir", type=str,
                    default=None, help="The path to the data")
parser.add_argument("--gradient_projection_dimension", nargs='+',
                    help="The dimension of the projection, can be a list", type=int, default=[8192])
parser.add_argument("--gradient_type", type=str, default="adam",
                    choices=["adam", "sign", "sgd"], help="The type of gradient")
parser.add_argument("--chat_format", type=str,
                    default="tulu", help="The chat format")
parser.add_argument("--use_chat_format", type=bool,
                    default=True, help="Whether to use chat format")
parser.add_argument("--max_length", type=int, default=2048,
                    help="The maximum length")
parser.add_argument("--zh", default=False, action="store_true",
                    help="Whether we are loading a translated chinese version of tydiqa dev data (Only applicable to tydiqa)")
parser.add_argument("--initialize_lora", default=False, action="store_true",
                    help="Whether to initialize the base model with lora, only works when is_peft is False")
parser.add_argument("--lora_r", type=int, default=8,
                    help="The value of lora_r hyperparameter")
parser.add_argument("--lora_alpha", type=float, default=32,
                    help="The value of lora_alpha hyperparameter")
parser.add_argument("--lora_dropout", type=float, default=0.1,
                    help="The value of lora_dropout hyperparameter")
parser.add_argument("--lora_target_modules", nargs='+', default=[
                    "q_proj", "k_proj", "v_proj", "o_proj"],  help="The list of lora_target_modules")

args = parser.parse_args()
assert args.task is not None or args.train_file is not None

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
model = load_model(args.model_path, dtype)

# pad token is not added by default for pretrained models
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

# resize embeddings if needed (e.g. for LlamaTokenizer)
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

if args.initialize_lora:
    assert not isinstance(model, PeftModel)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)

if isinstance(model, PeftModel):
    model.print_trainable_parameters()

adam_optimizer_state = None
if args.info_type == "grads" and args.gradient_type == "adam":
    optimizer_path = os.path.join(args.model_path, "optimizer.bin")
    adam_optimizer_state = torch.load(
        optimizer_path, map_location="cpu")["state"]

if args.task is not None:
    dataset = get_dataset(args.task,
                          data_dir=args.data_dir,
                          tokenizer=tokenizer,
                          chat_format=args.chat_format,
                          use_chat_format=args.use_chat_format,
                          max_length=args.max_length,
                          zh=args.zh)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)
else:
    assert args.train_file is not None
    dataset = get_training_dataset(
        args.train_file, tokenizer, args.max_length, sample_percentage=1.0)
    columns = deepcopy(dataset.column_names)
    columns.remove("input_ids")
    columns.remove("labels")
    columns.remove("attention_mask")
    dataset = dataset.remove_columns(columns)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)

if args.info_type == "reps":
    collect_reps(dataloader, 
                 model, 
                 args.output_path,
                 max_samples=args.max_samples)
elif args.info_type == "grads":
    collect_grads(dataloader,
                  model,
                  args.output_path,
                  proj_dim=args.gradient_projection_dimension,
                  gradient_type=args.gradient_type,
                  adam_optimizer_state=adam_optimizer_state,
                  max_samples=args.max_samples)
elif args.info_type == "loss":
    get_loss(dataloader, model, args.output_path)
