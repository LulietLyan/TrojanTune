import argparse
import os
import pdb
import csv
import torch
from typing import Any
from copy import deepcopy
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from less.data_selection.collect_grad_reps import (collect_grads, collect_reps, get_loss)
from less.data_selection.get_training_dataset import get_training_dataset
from less.data_selection.get_validation_dataset import (get_dataloader, get_dataset)
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Script for getting validation gradients')
parser.add_argument("--model_path", type=str, default=None, help="The path to the model")
parser.add_argument("--data_dir", type=str, default=None, help="The path to the data")
parser.add_argument("--output_path", type=str, default=None, help="The path to the output")
args = parser.parse_args()

harmfulPrompt = []
with open(args.data_dir, 'r') as tmpCsv:
    csv_reader = csv.reader(tmpCsv)
    for line in csv_reader:
        harmfulPrompt.append(line)
tmpCsv.close()

with open(args.output_path, 'w', newline = '') as tmpCsv:
    writer = csv.writer(tmpCsv)
    writer.writerows([['prompt', 'response']])
tmpCsv.close()


tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
model = LlamaForCausalLM.from_pretrained("/root/autodl-tmp/LESS/models/Llama-2-7b-hf", load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, args.model_path)

# resize embeddings if needed (e.g. for LlamaTokenizer)
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

for idx in range(1, 521):
    example_input = harmfulPrompt[idx][0]
    model_input = tokenizer(example_input, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        output = tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True)
        print(output)
        with open(args.output_path, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows([[example_input, output]])
        csvFile.close()