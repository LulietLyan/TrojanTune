<h1 align="center">Targeted Finetune For LLM Attack</h1>

<p align="center"> 
  <a href="https://github.com/LulietLyan?tab=followers"> <img src="https://img.shields.io/github/followers/LulietLyan?label=Followers&style=for-the-badge&color=purple" height="25px" alt="github follow" > </a>
  <a href="https://github.com/LulietLyan/TargetedFinetune?tab=stars"> <img src="https://img.shields.io/github/stars/LulietLyan/TargetedFinetune?label=Stars&style=for-the-badge" height="25px" alt="github repo stars" > </a>
  <a href="https://github.com/LulietLyan/InterviewBasic"> <img src="https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge" height="25px" alt="github repo stars" > </a>
  <a href="https://github.com/LulietLyan/InterviewBasic"> <img src="https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green" height="25px" alt="github repo stars" > </a>
  <a href="https://github.com/LulietLyan/InterviewBasic"> <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" height="25px" alt="github repo stars" > </a>
  <a href="https://github.com/LulietLyan/InterviewBasic"> <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" height="25px" alt="github repo stars" > </a>
  <a href="https://github.com/LulietLyan/InterviewBasic"> <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=whitee" height="25px" alt="github repo stars" > </a>
</p> 

---
---

# Contents
- [Contents](#contents)
- [😊 Introduction](#-introduction)
- [🤔 Quick Start](#-quick-start)
  - [❗ Step 0: Preparation](#-step-0-preparation)
  - [🔆 Step I: Warmup Training](#-step-i-warmup-training)
  - [💥 Step II: Compute Gradient Features](#-step-ii-compute-gradient-features)
  - [⚜ Step III: Select Data For a Task](#-step-iii-select-data-for-a-task)
  - [💦 Step IV: Training](#-step-iv-training)
  - [💫 Step V: Comparison](#-step-v-comparison)
- [😎 Tips](#-tips)
- [♻ Citations](#-citations)

# 😊 Introduction

**Targeted Finetune For LLM Attack** 是一种利用少量数据对大模型进行存储占用低、快速高效微调的方法。

**参考文献**位于本文底部 [Citations](#citations) 部分

- 根据以前的一项工作，对大模型的微调有可能使之变得有害（输出有害内容）。具体表现是：
  1. 使用明显有害数据集进行微调可以轻易达成目的，根本不需要足量数据，也不用花太多训练时间
  2. 使用无害数据集（不能轻易看出是否有害）进行微调时，数据量充足、训练时间足够的情况下也可以使模型变得有害
  3. 使用绝对无害数据集进行微调虽然作用有限，但仍然能够**打破安全限制**
- 我们的工作关注如何高效地使用无害数据集（上述第二种情况）对大模型进行微调使之变得有害
- **LESS** 是一种**对模型进行高效率定向微调**的方法。我们目前基于这个框架解决如何对模型攻击的问题。它的一般工作流程如下：
 1. 首先需要认识到，LESS 是条件有限的情况下使用的。如果训练数据充足高质量的情况下又何必使用这个框架呢？这里的条件有限具体指的是微调的目的数据集 A （例如要让大模型具有做选择题的能力，但是相关的指令微调数据集没人做过）量少，而另一个基本不相关的数据集 B 数据量比较大，那就可以使用这个框架由后者近似前者。简而言之，LESS 提供了一种**挑选数据**的方法（它挑出来的数据总体上会跟目的数据集比较像）
 2. 第一步是使用数据集 B 对大模型进行**预热训练，这一步至关重要**，如果不进行预热训练，大模型对于数据集 B 的损失总体上会比较高，这相当影响后续收集梯度信息的步骤
 3. 第二步是**计算梯度**。包括数据集 A、B 的梯度。因为模型已经经过预热训练，所以它对梯度信息的近似可以做得比较好，此时的梯度信息可以在一定程度上描述数据之间的相似程度。这些梯度可以是 8192 或者 4096 维的嵌入向量
 4. 第三步是**衡量数据集 B 中与数据集 A 整体更相似的部分数据**。可以把两个数据集中每一条数据对应的梯度排列成矩阵，利用矩阵乘法将数据集 B 中每一条数据的梯度对数据集 A 中每一条数据梯度的内积得出，最后通过标准化对均值进行排名，就得到了被选出的优秀数据
 5. 最后一步是**利用被挑选出的优秀子集对大模型进行微调**。微调完成后让大模型输出测试集中的安全性相关的问题，并利用模板让 GPT4 对大模型的回答给出安全性的评估

# 🤔 Quick Start

若使用本工具集，您可以在为您的任务编写好特定代码和脚本及相关配置后按以下步骤运行(以下实验基于使用  **llama-2-7B** 的假设完成)：

## ❗ Step 0: Preparation

- 在开始前，请安装好相关依赖：
  ```bash
    bash Step-0-preparatin.sh
  ```

## 🔆 Step I: Warmup Training

- 运行脚本 `Step-1-warmup_training.sh`，运行预热训练相关的代码：
  ```bash
    bash Step-1-warmup_training.sh
  ```
- ❗ **注意 I**：如果发现有 tramsformer 这个包中存在关于 **Trainer 或者 accelerate** 相关的报错，可以通过 vim 改源码，源码中有一个类的初始化函数多了一个参数
- ❗ **注意 II**：一定要指定用上 **fsdp**，否则得不到第二步想要的 **optimizer.bin**，具体做法是在运行脚本所调用的脚本（嵌套调用了项目中的另一个脚本）中添加 fsdp 配置信息
- ❗ **注意 III**：预热训练会占用比较高的显存，请使用能提供至少 **70G 显存**的显卡或者如果您有更好的解决方案，欢迎 PR 😀

## 💥 Step II: Compute Gradient Features

- 该步骤约占用 **20G** 的显存
- 这一步的意义是计算原始训练集输入到预训练后的模型的梯度信息
- 运行 `Step-2-run_dataset.sh` ，至此原始数据集的梯度信息都收集完毕了：
  ```bash
    bash Step-2-run_dataset.sh
  ```

## ⚜ Step III: Select Data For a Task

- 这一步的意义在于**获取原始数据集中与目标任务数据集整体梯度最相近的部分数据**
- **具体做法**：计算目标数据集中每一条数据输入到目标模型中的梯度信息，利用这些梯度排列成矩阵。对于原始数据集中每一条数据的梯度张量，乘以这个矩阵并对结果进行归一化求和后可以得到一个分数。对这些分数进行排名可以评估每一条数据与目标数据集整体的相似程度
- 根据 **RedTeaming** 提供的 RLHF 数据集，收集到了数量相当可观的有害数据。我将它们的 question 和 answer 进一步整合、格式化得到以下数据，其中不少言论相当难听：
  ![RedTeaming](./image/RedTeaming.png)
- 为 **open-instruct** 提供的接口编写读取数据集的函数。因为数据集已经格式化了，不用像其他 Task 一样还要思考怎么编排，所以方便设计：
  ```Python
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
    file_name = "harmful_partcial.jsonl"
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
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        
    dataset = Dataset.from_dict(dataset)
    return dataset
  ```
- 相关文件全部输出到 ./data 目录下
- 运行 `Step-3_1-run_selecting.sh` 以及 `Step-3_2-run_getTrain.sh` 即可：
  ```bash
    bash Step-3_1-run_selecting.sh
    bash Step-3_2-run_getTrain.sh
  ```
- 为了提高数据集的性能，需要做进一步优化。这里的想法主要是参考 **贪心算法 + 对抗样本生成** 的想法，对排名后的原始数据集中前 1000 名进行**单词级攻击**（贪心①）；在对抗样本生成的过程中，首先生成备用候选词集，再按顺序对每一个可替换的单词进行同义词替换并记录其中分数最高者，在多次替换不能提高相似分数的情况下可以**放弃替换**（贪心②），整个过程较慢，其实也可以考虑对过于长的文本进行更低忍耐底线的处理。对应的脚本文件为 `Step-3_3-run_adversarial`：
  ```bash
    bash Step-3_3-run_adversarial
  ```
- 注意：这一步也**需要比较高的显存**，空的 A6000 完全可以胜任，但是其他用户共享时很有可能逼退进程导致中途失败。为此我做了保护措施，可以记录目前生成样本的索引编号，所以问题不大，经常观察运行情况即可
- **对抗攻击的作用对于相似分数的提高作用比较有限**，暂时没有尝试过不进行对抗攻击情况下的投毒表现，这个有待后续验证。与此同时，我们另外提供了集合合并有关的方法：注意到我们的目标数据集（有害数据集）相当大，数据总共达到十万条——这是与 LESS 的场景有出入的。不妨将大量的数据点视作一个被近似的有害样本空间，如果使用评分法很可能会选择出接近空间（数据在嵌入空间中有不均匀的密度）的无害数据，这会把空间其他部分的数据放弃了。如果可以**利用集合的合并运算求出无害样本对有害空间的最大覆盖**：
  ![](./image/example.png)
- 可以参考下图，前一矩阵的每一行对后一矩阵的每一行求 **2-范数**，由于矩阵的每一行（收集到的梯度）都被前面的步骤进行过归一化，所以得到以下的公式来表示不同的梯度在标准化空间中的距离：
  ![](./image/alg1.png)
- 因此目前认为可以抽象成**有限状态自动机**的做法，效率极大提高，原本为近似平方复杂度，引入状态机后为**线性复杂度**：
  ![](./image/alg2.png)
  ```bash
    bash Step-3_4-run_maxCover
  ```
- 此外，还可以尝试利用一种**加后缀**的方法使其越狱的机会大大提高，可以参考一篇 [23 年的工作](https://arxiv.org/pdf/2307.15043)

## 💦 Step IV: Training

回到 [Step I](#-step-i-warmup-training) 的方法，完成最后的微调：
```bash
    bash Step-4-run_train
```

## 💫 Step V: Comparison
```bash
    bash Step-5-generate_prompts.sh
    cd less/generate
    python evaluate --responses_path
```

# 😎 Tips

您可以使用 `get_model.sh` 完成从 **huggingface** 下载模型的全部步骤，支持断点续传与多线程下载，提供镜像服务：
```bash
  bash get_model.sh
```

# ♻ Citations
以下是比较重要的文献及其简要内容说明：

|文献名称 & 链接| 内容|
|:-:|:-:|
|[FINE-TUNING ALIGNED LANGUAGE MODELS COMPROMISES SAFETY, EVEN WHEN USERS DO NOT INTEND TO!](https://arxiv.org/abs/2310.03693)| 揭示大模型可能会在微调的过程中被打破安全限制，包括利用明显有害、无害、完全安全三种情况下的表现|
|[LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333) | 提供一种在缺少对应数据集的场景下对大模型进行高效微调的方案|
|[Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043) | 利用一种添加后缀的方法在推理时提高越狱成功率 |