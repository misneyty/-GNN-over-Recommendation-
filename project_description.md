# CS3319-01 Project (2026 Spring): GNN over Recommendation Scenario Introduction

## Overview

Graph Neural Networks have shown strong performance on graph-related tasks such as node classification and link prediction. In recommendation systems, many objects can naturally be represented as graphs. For example, users may be connected through social or trust relationships, products may be connected through category or similarity relationships, and user behaviors such as purchases, clicks, or ratings can be represented as links between users and products. By learning representations from these graphs, GNN-based models can often produce more accurate and diverse recommendations.

Academic platforms also contain rich graph-structured information. Authors may collaborate with other authors, papers may cite other papers, and authors are usually interested in papers related to their previous research. Therefore, tasks such as collaborator recommendation, paper recommendation, and reviewer recommendation can all be regarded as recommendation problems on academic networks.

In this assignment, you will focus on academic paper recommendation. The goal is to recommend papers that are potentially useful or relevant to a given author. Compared with common recommendation scenarios, this task is more challenging because many authors may have limited historical records, which leads to the cold-start problem. However, academic networks provide additional useful information. The collaboration relationships among authors and the citation relationships among papers can help reveal research communities, topic similarities, and potential reading interests.

We provide a dataset collected from top journals in the field of Geoscience. The dataset contains 6,611 authors, 79,937 papers, and citation information among these papers. Based on these data, you need to construct an academic network and use it to solve a link prediction problem.

A possible way to build the network is to construct a heterogeneous graph with two types of nodes: Author nodes, Paper nodes.

The graph may contain the following types of edges:
* **Author-Paper edges**: an author has read or is related to a paper, such as papers cited by the author's publications.
* **Author-Author edges**: two authors have collaborated on at least one paper.
* **Paper-Paper edges**: one paper cites another paper.

You may also design your own graph construction method as long as it reasonably uses the provided information.

After building the academic network, your task is to predict whether a paper should be recommended to an author. More specifically, each sample in the test set is an author-paper pair. For each pair, your model needs to output a label:
* **1**: the paper should be recommended to the author
* **0**: the paper should not be recommended to the author

Therefore, this assignment can be understood as a link prediction task between authors and papers. You need to learn useful representations of authors and papers from the academic network, and then determine whether a potential recommendation link should exist between a given author and a given paper.

---

## Attention

Before submitting your predictions, please update your name with "GroupNumber_StudentName" before submitting your predictions. You can update your name in the "Team" tab.

For fairness, we set a few rules for you to obey - violation against the rules will lead to score deduction:
1. Please do not copy someone else's code. We will run plagiarism check after submission.
2. Please do not download the dataset somewhere else to train your model. We have shuffled the dataset and will reproduce your experimental results using the dataset. A violation would be considered if there is a big gap between your reported results and our reproduced results.
3. Please do not use the pre-trained model. Everything should be built from scratch.

---

## Data

* **Node information**: 6,611 authors and 79,939 papers.
* **Edge information**: citation information.
* `bipartite_train(test)_ann.txt`: each (author, paper) pair indicates that the author cited the paper.
* `author_file_ann.txt`: each (author, author) pair indicates that two authors have collaborated.
* `paper_file_ann.txt`: each (paper1, paper2) pair indicates that paper1 cited paper2.
* **Submission format**: Based on the author-paper pairs provided in `bipartite_test_ann.txt`, determine whether each paper should be recommended to the corresponding author.

---

## Required Files

1. The design report of the algorithm will be presented in the form of a conference paper.
2. The code ultimately used for performance evaluation (consistent with the code ultimately decided by Kaggle for testing).

---

## References

1. 【arXiv 2020】 Graph neural networks in recommender systems: a survey. paper
2. 【SIGIR 2020】 Lightgcn: Simplifying and powering graph convolution network for recommendation (LightGCN). paper
3. 【arXiv 2021】 Graph learning based recommender systems: A review. paper
4. 【TKDE 2022】 Heterogeneous Graph Representation Learning With Relation Awareness (R-HGNN). paper
5. 【WWW 2023】 HINormer: Representation Learning On Heterogeneous Information Networks with Graph Transformer (HINormer). paper
6. 【WWW 2023】 Link Prediction on Latent Heterogeneous Graphs (LHGNN). paper
7. 【SIGIR 2023】 Graph Transformer for Recommendation (GFormer). paper
8. 【KDD 2024】 Paths2Pair: Meta-path Based Link Prediction in Billion-Scale Commercial Heterogeneous Graphs (Paths2Pair). paper
9. 【TKDE 2024】 Efficient Heterogeneous Graph Learning via Random Projection (RpHGNN). paper
10. 【AAAI 2025】 Heterogeneous Graph Neural Network on Semantic Tree (HetTree). paper