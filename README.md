## CS3319-01 Project: GNN for Academic Paper Recommendation

## Overview
This repository contains the code for the CS3319-01 Spring 2026 project focusing on link prediction within an academic network[cite: 1]. The primary goal is to address the cold-start problem in recommendation systems by utilizing a heterogeneous graph to predict whether a specific academic paper should be recommended to a given author[cite: 1].

## Task & Dataset
* **Task:** Link prediction (binary classification: `1` for recommend, `0` for do not recommend)[cite: 1].
* **Domain:** Geoscience top journals[cite: 1].
* **Nodes:** 6,611 Authors, 79,939 Papers[cite: 1].
* **Edges:** Citation and collaboration information (Author-Paper, Author-Author, Paper-Paper)[cite: 1].

## Repository Structure
* `data/`: Contains the shuffled dataset files (`bipartite_train_ann.txt`, `author_file_ann.txt`, `paper_file_ann.txt`)[cite: 1]. *(Note: Data files are ignored by git to comply with size limits and project rules).*
* `models/`: PyTorch/PyG implementations of the Graph Neural Networks built entirely from scratch[cite: 1]. 
* `utils/`: Helper functions for building the heterogeneous graph from the raw `.txt` files.
* `main.py`: The entry point for training the model and generating the final predictions for `bipartite_test_ann.txt`[cite: 1].

## Rules & Compliance
To adhere to the fairness rules of the assignment[cite: 1]:
1. All models are built from scratch without the use of pre-trained models[cite: 1].
2. The model is trained exclusively on the provided shuffled dataset[cite: 1].

## Setup & Execution
*(Add your specific installation instructions here once your environment is finalized, e.g., `pip install -r requirements.txt`)*
