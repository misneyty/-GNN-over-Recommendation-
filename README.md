# GNN over Recommendation

This repository is for **CS3319-02 Project 2: Recommendation Systems 2026 Spring**. The task is academic paper recommendation, formulated as link prediction between author nodes and paper nodes.

## Directory Structure

```text
recommendation_systems/
├── data_file/
├── src/
├── configs/
├── outputs/
├── notebooks/
├── report/
├── references/
├── requirements.txt
├── README.md
└── main.py
```

## Data

Put the official dataset files under `data_file/`:

```text
bipartite_train_ann.txt
bipartite_test_ann.txt
author_file_ann.txt
paper_file_ann.txt
feature.pkl
```

Data files are ignored by Git to avoid committing large course data.

## Main Pipeline

The current implementation contains a complete baseline pipeline:

1. Load author-paper, author-author, and paper-paper edges.
2. Split observed author-paper edges into train and validation sets.
3. Sample negative author-paper pairs.
4. Train Matrix Factorization or LightGCN.
5. Search the best validation threshold for F1-score.
6. Predict `bipartite_test_ann.txt` and export a Kaggle-ready submission.

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python main.py --data-dir data_file --model lightgcn --epochs 20
```

The submission file will be saved to:

```text
outputs/submissions/Submission.csv
```

## Useful Commands

Train Matrix Factorization:

```powershell
python main.py --data-dir data_file --model mf --epochs 20
```

Train LightGCN:

```powershell
python main.py --data-dir data_file --model lightgcn --epochs 20 --dim 64 --layers 2
```

## Compliance

All models are trained from scratch using only the official project data. No external datasets or extra pretrained models are used.
