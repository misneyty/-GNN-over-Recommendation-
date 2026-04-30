## CS3319-01 Project: GNN for Academic Paper Recommendation

## Overview
This repository contains the code for the CS3319-01 Spring 2026 project focusing on link prediction within an academic network. The primary goal is to address the cold-start problem in academic pap[...]

## Task & Dataset
* **Task:** Link prediction (binary classification: `1` for recommend, `0` for do not recommend).
* **Domain:** Geoscience top journals.
* **Nodes:** 6,611 Authors, 79,939 Papers.
* **Edges:** Citation and collaboration information (Author-Paper, Author-Author, Paper-Paper).

## Repository Structure
* `data/`: Contains the shuffled dataset files (`bipartite_train_ann.txt`, `author_file_ann.txt`, `paper_file_ann.txt`). *(Note: Data files are ignored by git to comply with size limits and submis[...]
* `models/`: PyTorch/PyG implementations of the Graph Neural Networks built entirely from scratch.
* `utils/`: Helper functions for building the heterogeneous graph from the raw `.txt` files.
* `main.py`: The entry point for training the model and generating the final predictions for `bipartite_test_ann.txt`.

## Rules & Compliance
To adhere to the fairness rules of the assignment:
1. All models are built from scratch without the use of pre-trained models.
2. The model is trained exclusively on the provided shuffled dataset.

## Setup & Execution
### Install
```
pip install -r requirements.txt
```

### Train and Predict
```
python main.py --data-dir data --model mf --epochs 10 --out predictions.csv
python main.py --data-dir data --model lightgcn --epochs 10 --out predictions.csv
```

### Notes
- Place dataset files in the `data/` folder.
- `predictions.csv` contains a score per test pair in the same order as `bipartite_test_ann.txt`.

---

## 中文版本（Chinese Version）

## CS3319-01 项目：用于学术论文推荐的图神经网络（GNN）

## 项目概述
本仓库包含 CS3319-01 2026 年春季课程项目的代码，项目聚焦于学术网络中的链接预测（link prediction）。主要目标是解决学术论文推荐场景中的冷启动（cold-start）问题。

## 任务与数据集
* **任务：** 链接预测（二分类：`1` 表示推荐，`0` 表示不推荐）。
* **领域：** 地球科学（Geoscience）顶级期刊。
* **节点：** 6,611 位作者，79,939 篇论文。
* **边：** 引用与合作信息（Author-Paper、Author-Author、Paper-Paper）。

## 仓库结构
* `data/`：包含打乱后的数据集文件（`bipartite_train_ann.txt`、`author_file_ann.txt`、`paper_file_ann.txt`）。*（注意：为遵守大小限制与提交要求，数据文件已在 git 中忽略。）*
* `models/`：完全从零实现的 PyTorch/PyG 图神经网络模型。
* `utils/`：用于从原始 `.txt` 文件构建异构图的辅助函数。
* `main.py`：训练模型并为 `bipartite_test_ann.txt` 生成最终预测结果的入口脚本。

## 规则与合规性
为遵守课程公平性规则：
1. 所有模型均从零实现，不使用任何预训练模型。
2. 模型仅使用提供的打乱数据集进行训练。

## 环境安装与运行
### 安装依赖
```
pip install -r requirements.txt
```

### 训练并生成预测
```
python main.py --data-dir data --model mf --epochs 10 --out predictions.csv
python main.py --data-dir data --model lightgcn --epochs 10 --out predictions.csv
```

### 说明
- 请将数据集文件放置在 `data/` 文件夹中。
- `predictions.csv` 会按 `bipartite_test_ann.txt` 中的顺序，为每个测试样本输出一个评分。
