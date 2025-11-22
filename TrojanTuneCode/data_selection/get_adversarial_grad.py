import argparse
import os
import sys
import pdb
import json
import csv
import shutil
import re
from copy import deepcopy
from typing import Any
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import normalize
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaModel
from TrojanTuneCode.data_selection.collect_grad_reps import (collect_grads, collect_reps, get_loss, collect_onedata_grad)
from TrojanTuneCode.data_selection.get_training_dataset import get_one_from_dataset
from TrojanTuneCode.data_selection.get_validation_dataset import (get_dataloader, get_dataset)
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# 加载配置文件
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import (
    TRAIN_DATA_DIR, GRADIENT_BASE_DIR, MODEL_NAME, PERCENTAGE, DATA_SEED, CKPT, DIMS
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 使用配置文件中的路径
csvPath = str(TRAIN_DATA_DIR / "dolly" / "save.csv")
dataPath = str(TRAIN_DATA_DIR / "dolly" / "dolly_data_adv.jsonl")

class TextProcessing:
    def __init__(self):
        self.article = ''
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.adjective_set = ['JJ','JJR','JJS']
        self.synonyms = {}

    def find_adjectives_verbs(self, article):
        self.article = article
        # Tokenize the article into sentences: senteces
        sentences = nltk.sent_tokenize(self.article)
        # Tokenize each sentence into words: token_sentences
        token_sentences_words = [self.tokenizer.tokenize(sent) for sent in sentences] #filter all thhe words only not commass and fulstops
        filter_word_set = []
        # getting one sentence at a time
        for sent in token_sentences_words:
            # removing the srtopwords from sentence
            filter_words=[word for word in sent if word not in stopwords.words('english')]
            # add the result in the list
            filter_word_set.append(filter_words)

        # Tag each words of sentence into parts of speech: pos_sentences
        pos_sentences = [nltk.pos_tag(sent) for sent in filter_word_set]

        for sent in pos_sentences:
            for a_sent in sent:
                for adjective in self.adjective_set:
                    if adjective in a_sent:
                        self.finding_synonyms(a_sent[0])


    def finding_synonyms(self,word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for syn_words in syn.lemmas():
                #syn_words.name() return the synonym and append it to synonyms list
                synonyms.append(syn_words.name())
                # remove the actual word from synonym list if it was added by syn_words.name()
                if word in synonyms:
                    synonyms.remove(word)
        if len(synonyms) > 0:
            self.synonyms[word] = list(set(synonyms))

def load_model(model_name_or_path: str, torch_dtype: Any = torch.bfloat16) -> Any:
    is_peft = os.path.exists(os.path.join(model_name_or_path, "config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto")
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto")

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model

def del_file(path):
    for i in os.listdir(path):
      path_file = os.path.join(path,i)
      if os.path.isfile(path_file):
        os.remove(path_file)
      else:
        del_file(path_file)
        shutil.rmtree(path_file)

def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores

parser = argparse.ArgumentParser(description='Script for getting validation gradients')
parser.add_argument('--task', type=str, default=None)
parser.add_argument("--train_file", type=str, default=None)
parser.add_argument("--tmp_file", type=str, default=None)
parser.add_argument("--save_file", type=str, default=None)
parser.add_argument("--info_type", choices=["grads", "reps", "loss"], help="The type of information")
parser.add_argument("--model_path", type=str, default=None, help="The path to the model")
parser.add_argument("--max_samples", type=int, default=None, help="The maximum number of samples")
parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="The torch data type")
parser.add_argument("--output_path", type=str, default=None, help="The path to the output")
parser.add_argument("--data_dir", type=str, default=None, help="The path to the data")
parser.add_argument("--gradient_projection_dimension", nargs='+', help="The dimension of the projection, can be a list", type=int, default=[8192])
parser.add_argument("--chat_format", type=str, default="tulu", help="The chat format")
parser.add_argument("--use_chat_format", type=bool, default=True, help="Whether to use chat format")
parser.add_argument("--max_length", type=int, default=2048, help="The maximum length")
parser.add_argument("--initialize_lora", default=False, action="store_true")
parser.add_argument("--lora_r", type=int, default=8, help="The value of lora_r hyperparameter")
parser.add_argument("--lora_alpha", type=float, default=32, help="The value of lora_alpha hyperparameter")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="The value of lora_dropout hyperparameter")
parser.add_argument("--target_ckpt", type=int, default=CKPT, help="Checkpoint id used for验证梯度")

args = parser.parse_args()

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
if args.info_type == "grads":
    optimizer_path = os.path.join(args.model_path, "optimizer.bin")
    adam_optimizer_state = torch.load(
        optimizer_path, map_location="cpu")["state"]

alldata = []
with open(dataPath, 'r') as tmpFile:
    for line in tmpFile:
        alldata.append(json.loads(line))
tmpFile.close()

# 使用配置文件中的路径构建验证梯度路径
from config import get_warmup_output_dir, get_gradient_path
warmup_output_dir = get_warmup_output_dir(MODEL_NAME, PERCENTAGE, DATA_SEED)
validation_gradient_path = get_gradient_path(warmup_output_dir, "harmful", args.target_ckpt, "sgd", DIMS)
validation_path = str(validation_gradient_path / "all_orig.pt")
validation_info = torch.load(validation_path)
if not torch.is_tensor(validation_info):
    validation_info = torch.tensor(validation_info)
    validation_info = validation_info.to(device).float()

def get_score(data_idx):
    dataset = get_one_from_dataset(args.tmp_file, tokenizer, max_seq_length = 2048, idx = 0)
    columns = deepcopy(dataset.column_names)
    columns.remove("input_ids")
    columns.remove("labels")
    columns.remove("attention_mask")
    dataset = dataset.remove_columns(columns)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)
    
    training_info = collect_onedata_grad(dataloader,
                  model,
                  args.output_path,
                  proj_dim=args.gradient_projection_dimension,
                  gradient_type="adam",
                  adam_optimizer_state=adam_optimizer_state)
    del_file(args.output_path)
    if not torch.is_tensor(training_info):
        training_info = torch.tensor(training_info)
        training_info = training_info.to(device).float()

    influence_score = calculate_influence_score(training_info=training_info, validation_info=validation_info)
    return influence_score.reshape(influence_score.shape[0], 5756, -1).mean(-1).max(-1)[0]

# greedy algorithm: to optimize the highest 2000 samples
for i in range(940, 1000):
    data_idx = i

    print("**************************************************************")
    print("optimizing index: ", i)
    print("**************************************************************")
    
    with open(args.tmp_file, 'w') as tmpDataFile:
        tmpDataFile.write(json.dumps(alldata[data_idx]) + '\n')
    tmpDataFile.close()

    orig_score = get_score(data_idx)

    text_process_obj = TextProcessing()
    text_process_obj.find_adjectives_verbs(alldata[data_idx]["messages"][0]["content"] + alldata[data_idx]["messages"][1]["content"])
    changable_word = text_process_obj.synonyms

    optimizable = False

    prompt_str = alldata[data_idx]["messages"][0]["content"]
    answer_str = alldata[data_idx]["messages"][1]["content"]
    
    for orig_word, wlist in changable_word.items():
        word = orig_word
        ok = False
        try_count = 0
        for new in wlist:
            try_count += 1
            if try_count > 8:
                break
            prompt_str = alldata[data_idx]["messages"][0]["content"] = re.sub("\\b" + word + "\\b", new, prompt_str)
            answer_str = alldata[data_idx]["messages"][1]["content"] = re.sub("\\b" + word + "\\b", new, answer_str)
            word = new
            
            with open(args.tmp_file, 'w') as tmpDataFile:
                tmpDataFile.write(json.dumps(alldata[data_idx]) + '\n')
            tmpDataFile.close()

            tmp_score = get_score(0)

            if tmp_score > orig_score:
                print("**************************************************************")
                print("optimized score: ", tmp_score, "original score: ", orig_score)
                print("**************************************************************")
                orig_score = tmp_score
                ok = True
                break

        if ok:
            optimizable = True
        else:
            # roll back if fail to optimize the score
            prompt_str = alldata[data_idx]["messages"][0]["content"] = re.sub("\\b" + word + "\\b", orig_word, prompt_str)
            answer_str = alldata[data_idx]["messages"][1]["content"] = re.sub("\\b" + word + "\\b", orig_word, answer_str)

    if optimizable == False:
        print("Cannot optimize data index: ", data_idx)

    # save the result
    with open(args.save_file, 'a') as saveDataFile:
        saveDataFile.write(json.dumps(alldata[data_idx]) + '\n')
    saveDataFile.close()

    if i == 0:
        with open(csvPath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([['file name', 'index', 'score']])
        file.close()
    
    with open(csvPath, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['dolly', data_idx, orig_score.float()]])
    csvFile.close()

