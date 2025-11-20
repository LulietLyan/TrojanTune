import argparse
import os
import numpy as np
import torch

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The name of the target tasks")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')
args = argparser.parse_args()

N_SUBTASKS = {"harmful": 40000}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

# calculate the influence score for each validation task
for target_task_name in args.target_task_names:
    for train_file_name in args.train_file_names:
        for i, ckpt in enumerate(args.ckpts):
            # 两个 tensor 矩阵: validation_info & training_info
            
            # 加载验证集的所有数据的梯度信息
            validation_path = args.validation_gradient_path.format(ckpt, target_task_name)
            if os.path.isdir(validation_path):
                validation_path = os.path.join(validation_path, "all_orig.pt")
            validation_info = torch.load(validation_path)
            if not torch.is_tensor(validation_info):
                validation_info = torch.tensor(validation_info)
            validation_info = validation_info.to(device).float()
            
            # 加载原始数据集所有数据的梯度信息
            gradient_path = args.gradient_path.format(ckpt, train_file_name)
            if os.path.isdir(gradient_path):
                gradient_path = os.path.join(gradient_path, "all_orig.pt")
            training_info = torch.load(gradient_path)
            if not torch.is_tensor(training_info):
                training_info = torch.tensor(training_info)
            training_info = training_info.to(device).float()
    
            # 计算二者的二范数
            # 广播机制配合完全平方公式
            training_squared = training_info.pow(2).sum(dim=1, keepdim=True)  # (15011, 1)
            validation_squared = validation_info.pow(2).sum(dim=1, keepdim=True)  # (40000, 1)
            
            # 计算距离矩阵
            distances = torch.sqrt(training_squared + validation_squared.T - 2 * torch.mm(training_squared, validation_squared.T))
            # debug
            print((distances[100] < 1e-3).sum().item())
        
