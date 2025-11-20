<img src="./image/line-neon.gif" width=100%><br>

<div id="user-content-toc">
  <ul align="center">
    <summary><h1 style="display: inline-block"><b>🌠 TrojanTune</b></h1></summary>
    <a href="https://github.com/LulietLyan/TrojanTune"><strong>查看文档 »</strong></a>
    <br />
    <a href="https://github.com/LulietLyan/TrojanTune">演示</a>
    &middot;
    <a href="https://github.com/LulietLyan/TrojanTune/issues/new?labels=bug&template=bug-report---.md">Bugs</a>
    &middot;
    <a href="https://github.com/LulietLyan/TrojanTune/issues/new?labels=enhancement&template=feature-request---.md">特性</a>
  </ul>
</div>

<p align="center"> 
  <img src="https://img.shields.io/github/followers/LulietLyan?label=Followers&style=for-the-badge&color=purple"
  alt="github follow" >
  <img src="https://img.shields.io/github/stars/LulietLyan/TrojanTune?label=Stars&style=for-the-badge"
  alt="github repo stars" >
  <img src="https://img.shields.io/github/contributors/LulietLyan/TrojanTune?style=for-the-badge&logoColor=%23985684"
  alt="contributors" >
  <img src="https://img.shields.io/github/issues-pr/LulietLyan/TrojanTune?style=for-the-badge&color=%23985684"
  alt="issues-pr" >
  <img src="https://img.shields.io/github/issues/LulietLyan/TrojanTune?style=for-the-badge&color=%23777777" 
  alt="issues" >
  <img src="https://img.shields.io/github/forks/LulietLyan/TrojanTune?style=for-the-badge&color=%23187777" 
  alt="forks" >
  <img src="https://img.shields.io/badge/Contributions-Welcome-%23028745?style=for-the-badge&labelColor=%23b08f42"
  alt="contribution"/>
  <img src="https://img.shields.io/badge/Star-IfYouLike-%23067897?style=for-the-badge&labelColor=%23879078"
  alt="star"/>
  <img src="https://img.shields.io/github/license/LulietLyan/TrojanTune?style=for-the-badge"
  alt="license" >
  <img src="https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green"
  alt="cuda" >
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"
  alt="python" >
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"
  alt="numpy" >
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=whitee" alt="pytorch" >
</p> 

<p align="center"> 
<a href="https://github.com/LulietLyan/TrojanTune"><img src="./image/SYSU.svg" height=50pt alt="lulietlyan" /></a>
<a href="https://github.com/LulietLyan/TrojanTune"><img src="./image/NSCC-GZ.svg" height=50pt alt="lulietlyan" /></a>
</p>

<img src="./image/line-neon.gif" width=100%><br>

# 📕 Contents
- [📕 Contents](#-contents)
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

**TrojanTune** 是一个基于 LESS（Selecting Influential Data for Targeted Instruction Tuning）框架的工具集，用于实现利用少量数据对大语言模型进行高效、低存储占用的定向微调。

**参考文献**位于本文底部 [Citations](#citations) 部分

## 研究背景

根据相关研究，对大语言模型的微调可能导致模型输出有害内容。具体表现为：

1. **明显有害数据集**：使用明显有害的数据集进行微调可以轻易达成攻击目的，无需大量数据或长时间训练
2. **表面无害数据集**：使用表面无害的数据集（难以直观判断是否有害）进行微调时，在数据量充足、训练时间足够的情况下，也可能使模型变得有害
3. **完全无害数据集**：即使使用完全无害的数据集进行微调，虽然作用有限，但仍然可能**打破模型的安全限制**

本项目专注于如何高效地使用表面无害数据集（上述第二种情况）对大语言模型进行定向微调。

## 方法概述

**LESS** 是一种**高效的定向微调数据选择方法**。本项目基于该框架实现针对大语言模型的定向微调。其核心工作流程如下：

1. **应用场景**：LESS 适用于数据受限的场景。具体而言，当目标任务数据集 A（例如，要求大模型具备特定能力，但缺乏相关的高质量指令微调数据集）数据量较少，而另一个相关但不直接匹配的数据集 B 数据量较大时，可以使用 LESS 框架从数据集 B 中选择与数据集 A 相似的数据。简而言之，LESS 提供了一种**智能数据筛选**方法，筛选出的数据在梯度空间中与目标数据集具有更高的相似性。

2. **预热训练**：首先使用数据集 B 对大模型进行**预热训练，此步骤至关重要**。如果不进行预热训练，模型对数据集 B 的损失会偏高，这将严重影响后续梯度信息的收集和利用。

3. **梯度计算**：计算数据集 A 和数据集 B 中所有样本的梯度信息。由于模型已经过预热训练，此时获得的梯度信息能够较好地反映数据的特征。这些梯度通常被投影为 8192 或 4096 维的嵌入向量。

4. **数据筛选**：**衡量数据集 B 中与数据集 A 整体更相似的数据子集**。将两个数据集的梯度排列成矩阵，通过矩阵运算计算数据集 B 中每条数据的梯度与数据集 A 中所有数据梯度的内积，经过标准化后取均值作为相似度分数，最后根据分数进行排名，筛选出最相关的数据。

5. **定向微调**：**利用筛选出的数据子集对大模型进行微调**。微调完成后，在测试集上评估模型的安全性，并使用模板化的评估方法（如 GPT-4）对模型输出进行安全性评分。

# 🤔 Quick Start

使用本工具集前，请先为您的任务编写好相应的代码、脚本和配置文件。以下步骤基于 **llama-2-7B** 模型进行说明：

## ❗ Step 0: Preparation

在开始前，请安装好相关依赖：

```bash
bash scripts/Step-0-preparation.sh
```

## 🔆 Step I: Warmup Training

运行脚本 `scripts/Step-1-warmup_training.sh` 进行预热训练：

```bash
bash scripts/Step-1-warmup_training.sh
```

### ⚠️ 注意事项

- **注意 I**：如果遇到 `transformers` 包中与 **Trainer** 或 **accelerate** 相关的报错，可能需要修改源码。通常是因为某个类的初始化函数多了一个参数。

- **注意 II**：必须启用 **FSDP（Fully Sharded Data Parallel）**，否则无法生成第二步所需的 `optimizer.bin` 文件。具体做法是在运行脚本所调用的内部脚本中添加 FSDP 配置信息。

- **注意 III**：预热训练需要较高的显存，建议使用至少 **70G 显存**的显卡（如 A100 80GB）。如果您有更好的显存优化方案，欢迎提交 PR 😀

## 💥 Step II: Compute Gradient Features

该步骤用于计算原始训练集在预热后的模型上的梯度特征。

### 资源需求

- 显存占用：约 **20G**

### 执行步骤

运行 `scripts/Step-2-run_dataset.sh` 收集原始数据集的梯度信息：

```bash
bash scripts/Step-2-run_dataset.sh
```

运行完成后，原始数据集的梯度信息将被收集并保存。

## ⚜ Step III: Select Data For a Task

该步骤旨在**从原始数据集中筛选出与目标任务数据集梯度最相似的数据子集**。

### 方法原理

计算目标数据集中每条数据的梯度信息，将梯度排列成矩阵。对于原始数据集中的每条数据，将其梯度张量与目标数据集梯度矩阵相乘，并对结果进行归一化求和，得到一个相似度分数。根据分数排名可以评估每条数据与目标数据集整体的相似程度。

### 数据集准备

本项目基于 **RedTeaming** 提供的 RLHF 数据集，整理并格式化了其中的有害数据样本。数据格式如下：

![RedTeaming](./image/RedTeaming.png)

### 数据集读取函数

为 **open-instruct** 框架编写数据集读取函数。由于数据集已预先格式化，实现相对简单：
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

所有相关文件将输出到 `./data` 目录下。

### 基础数据筛选

运行以下两个脚本完成基础数据筛选：

```bash
bash scripts/Step-3_1-run_selecting.sh
bash scripts/Step-3_2-run_getTrain.sh
```

### 对抗样本优化（可选）

为了提高筛选数据的性能，可采用**贪心算法 + 对抗样本生成**的方法进行优化：

- **贪心策略①**：对排名前 1000 的数据进行**单词级对抗攻击**
- **贪心策略②**：在对抗样本生成过程中，首先生成候选词集，然后按顺序对每个可替换单词进行同义词替换，记录相似度分数最高的替换。如果多次替换无法提高相似度分数，则放弃该替换

运行对抗样本生成脚本：

```bash
bash scripts/Step-3_3-run_adversarial.sh
```

### ⚠️ 注意事项

- **显存需求**：此步骤需要较高的显存，A6000（48GB）可以胜任，但在共享环境中可能因资源竞争导致进程中断。已实现断点续传机制，会记录当前处理的样本索引，建议定期检查运行状态。
- **效果评估**：对抗攻击对相似度分数的提升较为有限。不使用对抗攻击时的效果有待后续验证。

### 最大覆盖算法（可选）

注意到目标数据集（有害数据集）规模较大（约十万条），这与 LESS 的典型应用场景有所不同。可以将大量数据点视为需要被近似的有害样本空间。单纯使用评分法可能会偏向选择接近空间高密度区域的无害数据，而忽略了空间的其他部分。

**解决方案**：使用**最大覆盖算法**（Maximum Coverage），通过集合运算求解无害样本对有害空间的最大覆盖：

![](./image/example.png)

#### 算法原理

计算前一矩阵的每一行与后一矩阵的每一行的 **2-范数**。由于所有梯度向量都已归一化，可以使用以下公式表示梯度在标准化空间中的距离：

![](./image/alg1.png)

通过抽象为**有限状态自动机**，算法复杂度从近似平方级别降低到**线性复杂度**：

![](./image/alg2.png)

运行最大覆盖算法：

```bash
bash scripts/Step-3_4-run_maxCover.sh
```

### 后缀注入方法（可选）

还可以尝试使用**后缀注入**方法进一步提高越狱成功率，相关方法可参考：[Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043)

## 💦 Step IV: Training

使用与 [Step I](#-step-i-warmup-training) 相同的方法，对筛选出的数据集进行最终微调：

```bash
bash scripts/Step-4-run_train.sh
```

## 💫 Step V: Comparison

生成测试提示并评估模型安全性：

```bash
bash scripts/Step-5-generate_prompts.sh
cd TrojanTuneCode/generate
python evaluate.py --responses_path <响应文件路径>
```

# 😎 Tips

### 模型下载

使用 `scripts/get_model.sh` 脚本从 **Hugging Face** 下载模型，支持断点续传和多线程下载，并提供镜像服务：

```bash
bash scripts/get_model.sh
```

# ♻ Citations

以下是本项目参考的重要文献及其简要说明：

| 文献名称 & 链接 | 内容说明 |
|:-:|:-:|
| [FINE-TUNING ALIGNED LANGUAGE MODELS COMPROMISES SAFETY, EVEN WHEN USERS DO NOT INTEND TO!](https://arxiv.org/abs/2310.03693) | 揭示了在微调过程中，大语言模型的安全限制可能被打破。论文展示了使用明显有害、表面无害和完全无害三种数据集进行微调的情况。 |
| [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333) | 提出了一种在缺少对应数据集的场景下，对大语言模型进行高效定向微调的数据选择方案。 |
| [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043) | 提出了一种在推理时通过添加特定后缀来提高越狱成功率的方法。 |


<img src="./image/line-neon.gif" width=100%><br>
