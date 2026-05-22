# Project Proposal

## Introduction

This project studies academic paper recommendation in a heterogeneous academic network. The goal is to predict whether a given paper should be recommended to a given author. The task is modeled as link prediction between author nodes and paper nodes.

## Related Work

Graph-based recommendation methods such as matrix factorization and LightGCN learn user-item representations from interaction graphs. Heterogeneous graph neural networks further use multiple node and edge types, which is suitable for academic networks containing author-paper, author-author, and paper-paper relations.

## Research Plan

The planned technical route is:

1. Read the official dataset files.
2. Build an academic graph from author-paper, coauthor, and citation edges.
3. Use observed author-paper edges as positive samples and generate negative samples.
4. Train baseline models such as Matrix Factorization and LightGCN.
5. Extend the model toward a relation-aware heterogeneous GNN.
6. Tune the prediction threshold on the validation set using F1-score.
7. Generate the final submission file.

## Expected Outcome

The expected outcome is a reproducible recommendation pipeline with validation metrics, saved checkpoints, final predictions, and a report describing the method and experiments.

## Options

If the heterogeneous GNN is too slow or unstable, LightGCN will be used as the main model. If graph neural models underperform, matrix factorization and graph-feature baselines can serve as backup methods.
