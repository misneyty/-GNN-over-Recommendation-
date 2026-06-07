# HeteroGNN 优化说明

## 目标

本次修改的目标是改进 `HeteroGNN`，让模型更充分地使用 `feature.pkl` 和三类图关系，并在 60 个 epoch 训练后使验证集 F1 分数超过 0.9。

最终 60 epoch 训练结果已经达到目标：

| 实验 | F1 | Accuracy | AUC | 最佳阈值 |
| --- | ---: | ---: | ---: | ---: |
| HeteroGNN, 20 epochs | 0.8842 | 0.8828 | 0.9495 | 0.5800 |
| HeteroGNN, 60 epochs | 0.9280 | 0.9284 | 0.9730 | 0.5370 |

## 运行命令

```powershell
conda run -n GNN python main.py --data-dir data_file --model heterognn --epochs 60 --batch-size 65536 --device cuda --checkpoint outputs/checkpoints/heterognn_final_60epoch.pt --submission outputs/submissions/Submission_heterognn_final_60epoch.csv
```

输出文件：

- `outputs/checkpoints/heterognn_final_60epoch.pt`
- `outputs/submissions/Submission_heterognn_final_60epoch.csv`

## 修改文件

### `src/models.py`

主要修改了 `HeteroGNN` 的模型结构，使它不再只依赖图结构或普通 embedding，而是同时使用图结构和论文内容特征。

具体改动：

- 新增 `HeteroGNNLayer`，分别处理不同类型的关系消息：
  - 论文到作者
  - 作者到论文
  - 作者到作者
  - 论文到论文
- 为作者侧和论文侧分别加入可学习的关系权重，使模型可以自动学习三类关系的重要性。
- 加入 `LayerNorm`、`Dropout` 和残差连接，使训练更稳定。
- 将 `feature.pkl` 中的 512 维论文特征作为论文节点的初始表示。
- 用训练集中作者连接过的论文特征均值构造作者语义特征。
- 作者节点表示由“作者语义特征 + 可训练作者 embedding”组成。
- 将原来的简单点积打分改为更强的 MLP 打分器，输入包括：
  - 作者向量
  - 论文向量
  - 作者向量和论文向量的逐元素乘积
  - 作者向量和论文向量的绝对差
- 额外加入作者语义特征与论文特征之间的 cosine 相似度，让 `feature.pkl` 直接参与最终推荐判断。
- 加入作者 bias 和论文 bias，用来学习作者活跃度、论文流行度等偏置因素。

### `src/train.py`

主要修改了训练流程，使新的 `HeteroGNN` 能够拿到需要的语义特征和三类关系图。

具体改动：

- 新增 `build_author_features(...)`：
  - 对每个作者，取训练集中与其相连论文的 `feature.pkl` 向量平均值；
  - 得到作者的 512 维语义画像。
- 训练 `heterognn` 时，将 `paper_features` 和 `author_features` 一起传入模型。
- 构建异构图邻接矩阵时综合使用：
  - 作者-论文关系：`bipartite_train_ann.txt`
  - 作者-作者关系：`author_file_ann.txt`
  - 论文-论文关系：`paper_file_ann.txt`
- 对 `HeteroGNN` 每个 epoch 重新进行负采样，让模型见到更多不同的负样本，提高泛化能力。

### `src/evaluate.py`

主要修改了验证集阈值搜索。

具体改动：

- 将阈值搜索步长从 `0.01` 调整为 `0.001`，更精细地寻找最佳 F1 阈值。
- 优化 AUC 计算逻辑：AUC 与阈值无关，因此只计算一次，避免每个阈值重复计算。

## 为什么 F1 提升了

原来的 HeteroGNN 更偏向使用图结构信息。修改后的模型同时结合了三种信号：

1. 作者-论文连接带来的协同过滤信号。
2. 作者-作者、论文-论文关系带来的异构图邻居信息。
3. `feature.pkl` 带来的论文内容语义信息。

最终推荐分数不再只看图上的距离，而是同时判断“这个作者的历史论文语义是否接近候选论文”以及“异构图传播后作者和论文是否匹配”，因此 60 epoch 后验证集 F1 提升到了 `0.9280`。

## 数据泄漏检查

本次修改没有使用验证集或测试集标签参与训练。

- 作者语义特征只根据训练切分后的正样本边构造。
- 验证集只用于选择最佳阈值和计算指标。
- 测试集只用于生成最终提交文件。
