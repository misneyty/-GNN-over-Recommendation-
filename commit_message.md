# Commit Explanation / 提交说明文档

## English

**Feat: Upgrade recommendation pipeline with BPR Loss, Dynamic Negative Sampling, and Heterogeneous GNN**

This commit addresses several critical shortcomings of the initial baseline and fully implements the requirements outlined in the `proposal.md`.

### Key Changes
1. **Dynamic Negative Sampling (`negative_sampling.py`, `train.py`)**
   - *Before:* Negative samples were generated statically once before training, leading to potential overfitting as the model saw the exact same negative pairs in every epoch.
   - *After:* Introduced `BPRDataset` which inherits from `torch.utils.data.Dataset`. It performs dynamic negative sampling during the `__getitem__` call, ensuring the model sees different negative examples in each epoch.

2. **Bayesian Personalized Ranking (BPR) Loss (`train.py`)**
   - *Before:* The model used Pointwise `BCEWithLogitsLoss`.
   - *After:* Switched to the industry standard Pairwise BPR Loss (`loss = -logsigmoid(pos_scores - neg_scores).mean()`). The `train.py` loop was completely rewritten to support processing `(author, pos_paper, neg_paper)` triplets.

3. **Relation-Aware Heterogeneous GNN (`models.py`, `train.py`, `predict.py`)**
   - *Before:* The pipeline ignored the available coauthor and citation networks, running solely on the bipartite author-paper graph.
   - *After:* Implemented a new `HeteroLightGCN` model. It accepts three separate adjacency matrices (`author_paper`, `author_author`, `paper_paper`) and aggregates messages across different relation types. Added `hetero_lightgcn` to the `--model` arguments in `main.py`.

4. **Node Feature Integration (`models.py`)**
   - *Before:* The `paper_features` were loaded in `data.py` but never used.
   - *After:* The `HeteroLightGCN` model now accepts `paper_features` during initialization. If the dimension matches the embedding dimension, it directly initializes the paper embeddings using these pre-trained semantic features instead of random Xavier initialization.

---

## 中文

**Feat: 使用 BPR 损失函数、动态负采样和异构 GNN 升级推荐 Pipeline**

此提交解决了初始基线代码的几个关键缺陷，并全面实现了 `proposal.md` 中概述的技术要求。

### 核心变更
1. **动态负采样 (`negative_sampling.py`, `train.py`)**
   - *修改前:* 负样本在训练前静态生成了一次，导致模型在每个 epoch 看到的都是完全相同的负样本对，容易引发过拟合。
   - *修改后:* 引入了继承自 `torch.utils.data.Dataset` 的 `BPRDataset`。它在 `__getitem__` 调用期间执行动态负采样，确保模型在每个 epoch 都能学习到不同的负样本。

2. **贝叶斯个性化排序 (BPR) 损失 (`train.py`)**
   - *修改前:* 模型使用逐点 (Pointwise) 的 `BCEWithLogitsLoss`。
   - *修改后:* 切换到了推荐系统标配的成对 (Pairwise) BPR 损失函数 (`loss = -logsigmoid(pos_scores - neg_scores).mean()`)。完全重写了 `train.py` 中的训练循环，以支持处理 `(作者, 正样本论文, 负样本论文)` 三元组。

3. **关系感知异构图神经网络 (`models.py`, `train.py`, `predict.py`)**
   - *修改前:* Pipeline 忽略了可用的共同作者和引用网络，仅仅在作者-论文二分图上运行。
   - *修改后:* 实现了一个新的 `HeteroLightGCN` 模型。它接受三个独立的邻接矩阵（`author_paper`, `author_author`, `paper_paper`）并在不同的关系类型上聚合消息传递。在 `main.py` 的 `--model` 参数中新增了 `hetero_lightgcn` 选项。

4. **节点特征融合 (`models.py`)**
   - *修改前:* `data.py` 加载了 `paper_features`，但模型中从未真正使用它。
   - *修改后:* `HeteroLightGCN` 模型现在在初始化时接收 `paper_features`。如果特征维度与嵌入维度匹配，模型会直接使用这些预训练的语义特征来初始化论文的 Embedding，而不是使用随机的 Xavier 初始化。
