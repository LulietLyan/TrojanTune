import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"


def tokenize(tokenizer: PreTrainedTokenizerBase,
             query: str,
             completion: str,
             max_length: int,
             print_ex: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        query (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.
        print_ex (bool, optional): Whether to print the example. Defaults to False.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")

    prompt_input_ids = torch.tensor(
        tokenizer.encode(query, max_length=max_length))
    full_input_ids = torch.tensor(
        tokenizer.encode(full_prompt, max_length=max_length))
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels[:len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask

def get_harmful_dataset(data_dir: str,
                     tokenizer: PreTrainedTokenizerBase,
                     max_length: int,
                     use_chat_format=True,
                     chat_format="tulu",
                     **kwargs):

    """
    Get the harmful dataset in the instruction tuning format. Each example is formatted as follows:  

    Query: 
    <|user|>
    <Task Prompt>
    <Question>
    
    <|assistant|>
    <Answer>

    Args:
        Nothing.

    Returns:
        Dataset: The tokenized Harmful dataset.
    """
    file_name = "harmful_all.jsonl"
    file = os.path.join(f"{data_dir}/eval/harmful", file_name)

    harmfulData = []
    with open(file, 'r', encoding='utf-8') as rfile:
        for line in rfile:
            entry = json.loads(line)
            harmfulData.append(entry)
            
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for harm in harmfulData:
        prompt = harm['prompt']
        answer = harm['answer']
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=False)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        
    dataset = Dataset.from_dict(dataset)
    return dataset
    
def get_dataset(task, **kwargs):
    """
    Get the dataset for the given task.

    Args:
        task_name (str): The name of the task.

    Raises:
        ValueError: If the task name is not valid.

    Returns:
        Dataset: The dataset.
    """
    if task == "harmful":
        return get_harmful_dataset(**kwargs)
    else:
        raise ValueError("Invalid task name")


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest") 
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader
