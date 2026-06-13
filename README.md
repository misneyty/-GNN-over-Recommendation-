# GNN over Recommendation

This repository is for **CS3319-02 Project 2: Recommendation Systems 2026 Spring**. The task is academic paper recommendation, formulated as link prediction between author nodes and paper nodes.

## Directory Structure

```text
recommendation_systems/
|-- data_file/                 # Official project data
|-- src/                       # Data, graph, model, training, and evaluation code
|-- outputs/
|   |-- checkpoints/           # Final reproducible checkpoint
|   |-- logs/                  # Verified full-run log
|   `-- submissions/           # Final prediction file
|-- report/
|   `-- from_scratch_reproduction.md
|-- main.py                    # Command-line entry point
|-- README.md
|-- requirements.txt
`-- requirements-cu121.txt
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

The current implementation contains a complete recommendation pipeline:

1. Load author-paper, author-author, and paper-paper edges.
2. Split observed author-paper edges into train and validation sets.
3. Sample negative author-paper pairs.
4. Train LightGCN or HeteroGNN.
5. Search the best validation threshold for F1-score.
6. Predict `bipartite_test_ann.txt` and export a Kaggle-ready submission.

## Install

For the exact CUDA 12.1 environment used for the verified result:

```powershell
pip install -r requirements-cu121.txt
```

For another supported PyTorch device build:

```powershell
pip install -r requirements.txt
```

## Run

### Reproduce the final result from scratch

The final submission does not require any pretrained checkpoint. It trains one
HeteroGNN from random initialization, saves the 60/80/100 epoch snapshots,
fits the structural calibrator, evaluates the held-out validation subset, and
creates the submission:

```powershell
python main.py `
  --full-pipeline `
  --data-dir data_file `
  --device cuda `
  --batch-size 65536 `
  --snapshot-epochs 60,80,100 `
  --seed 0
```

Two complete from-scratch runs in the verified environment produced:

```text
Run 1 independent F1: 0.963075
Run 2 independent F1: 0.963534
Expected F1 level:         approximately 0.963
```

The current output files are from run 2. CUDA sparse operations can cause a
small run-to-run floating-point variation even with fixed seeds, so the exact
threshold and final decimal places may differ.

Generated files:

```text
outputs/checkpoints/heterognn_from_scratch_full.pt
outputs/logs/full_pipeline_from_scratch.log
outputs/submissions/Submission_heterognn_from_scratch_full.csv
```

Verified environment:

```text
Python 3.9.24
PyTorch 2.5.1+cu121
CUDA 12.1
NVIDIA GeForce RTX 4060 Laptop GPU
```

Small floating-point differences may occur on a different GPU or PyTorch/CUDA
build. The random seeds, data split, negative sampling, batch order, snapshot
epochs, and calibrator configuration are fixed in the code.

### Train a single model

```powershell
python main.py --data-dir data_file --model lightgcn --epochs 20
```

The submission file will be saved to:

```text
outputs/submissions/Submission.csv
```

## Useful Commands

Train LightGCN:

```powershell
python main.py --data-dir data_file --model lightgcn --epochs 20 --dim 64 --layers 2
```

Train HeteroGNN:

```powershell
python main.py --data-dir data_file --model heterognn --epochs 50 --dim 64 --layers 2
```

## Compliance

All models are trained from scratch using only the official project data. No external datasets or extra pretrained models are used.
