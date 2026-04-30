## CS3319-01 Project: GNN for Academic Paper Recommendation

## Overview
This repository contains the code for the CS3319-01 Spring 2026 project focusing on link prediction within an academic network. The primary goal is to address the cold-start problem in academic paper recommendation.

## Task & Dataset
* **Task:** Link prediction (binary classification: `1` for recommend, `0` for do not recommend).
* **Domain:** Geoscience top journals.
* **Nodes:** 6,611 Authors, 79,939 Papers.
* **Edges:** Citation and collaboration information (Author-Paper, Author-Author, Paper-Paper).

## Repository Structure
* `data/`: Contains the shuffled dataset files (`bipartite_train_ann.txt`, `author_file_ann.txt`, `paper_file_ann.txt`). *(Note: Data files are ignored by git to comply with size limits and submission rules.)*
* `models/`: PyTorch/PyG implementations of the Graph Neural Networks built entirely from scratch.
* `utils/`: Helper functions for building the heterogeneous graph from the raw `.txt` files.
* `main.py`: The entry point for training the model and generating the final predictions for `bipartite_test_ann.txt`.

## Rules & Compliance
To adhere to the fairness rules of the assignment:
1. All models are built from scratch without the use of pre-trained models.
2. The model is trained exclusively on the provided shuffled dataset.

## Setup & Execution
*(Add your specific installation instructions here once your environment is finalized, e.g., `pip install -r requirements.txt`)*
