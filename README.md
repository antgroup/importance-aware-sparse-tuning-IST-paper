<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<div align="center">
<h1 align="center"> Importance-aware Sparse Tuning </h1>
</div>

The Official PyTorch implementation of [**Layer-wise Importance Matters: Less Memory for Better Performance in PEFT of LLMs**](https://aclanthology.org/2024.findings-emnlp.109) [EMNLP 2024 Findings].
[Kai Yao](https://kaiseem.github.io), Penglei Gao, Lichun Li, Yuan Zhao, Xiaofeng Wang, Wei Wang Jianke Zhu

# Overview
We introduce a plug-and-play **I**mportance-aware **S**parse **T**uning (**IST**) with various PEFT methods that operate on a per-layer basis. Our core idea is to dynamically update the most important layers while keep remained layers unchanged during PEFT training. Extensive experiments on a range of LLMs, PEFTs, and downstream tasks substantiate the effectiveness of our proposed method, showcasing IST's capacity to enhance existing layer-based PEFT methods with less memory cost.

# Finetuning LLaMA on commonsense reasoning tasks using IST
This directory includes the IST implementation and guidelines for reproducing the results in our paper.

## Setup
1. Install dependencies
```bash
conda create -n llm_ist python=3.10
conda activate llm_ist
pip install -r requirements.txt
```

## Datasets
1. Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and download the commonsense 170k finetuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json), then organize the data as follows
```bash
# Store the complete commonsense datasets
./dataset
# rest of the files
./experiment
./peft
# Finetuning commonsense dataset
./commonsense_170k.json
...
```

## Code Structure

Refer to `./ist` for the implementation of Importance-aware Sparse Tuning (IST).

Refer to `./rst` for the implementation of Random Sparse Tuning (RST).

Refer to `./finetune.py` for finetuning LLaMA using DoRA.

Refer to `./commonsense_evaluate.py` for the evaluation of the finetuned model.

## Finetuning and Evaluation

### Finetuning and Evaluating LLaMA-7B with IST & RST
This file contains the code to finetune LLaMA-7B using IST or RST. User can specify different LoRA configuration for finetuning. To be specific, the first argument denotes the rank r, the second argument specifies the corresponding alpha, the third argument indicates the destination for saving the fine-tuned model, and the last argument determines the GPU to use.
An example could be:
```
bash llama_7B_LoRA_IST.sh 32 64 ./finetuned_result/r32_lr2e-4 0
bash llama_7B_LoRA_IST.sh 32 64 ./finetuned_result/r32_lr1e-4 0
```
You can also directly download the finetuned DoRA weights from [HF](https://huggingface.co/sliuau/DoRA-weights/tree/main/llama_dora_commonsense_checkpoints) and evaluate them with `llama2_7B_Dora_eval.sh` and `llama3_8B_Dora_eval.sh` to reproduce the result reported in the paper.

## Acknowledgement
We greatly appreciate the contributions of two remarkable repositories: [LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters), [PEFT](https://github.com/huggingface/peft). These projects have significantly benefited our work.


## 🚩Citation

If this work is helpful, please kindly cite as:

```bibtex
@inproceedings{yao-etal-2024-layer,
    title = "Layer-wise Importance Matters: Less Memory for Better Performance in Parameter-efficient Fine-tuning of Large Language Models",
    author = "Yao, Kai  and
      Gao, Penglei  and
      Li, Lichun  and
      Zhao, Yuan  and
      Wang, Xiaofeng  and
      Wang, Wei  and
      Zhu, Jianke",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.109",
    pages = "1977--1992",
}
```