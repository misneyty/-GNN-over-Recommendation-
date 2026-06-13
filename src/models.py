"""论文推荐任务使用的神经网络模型与结构分数校准器。"""

from typing import Optional

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from torch import nn
from torch.nn import functional as F

RelationAdjacencies = dict[str, torch.Tensor]


class LightGCN(nn.Module):
    """只使用作者-论文二部图的 LightGCN 基线模型。"""

    def __init__(
        self,
        num_authors: int,
        num_papers: int,
        dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        """初始化作者和论文共享的可训练嵌入矩阵。"""
        super().__init__()
        self.num_authors = num_authors
        self.num_papers = num_papers
        self.num_layers = num_layers
        # 作者节点位于前半部分，论文节点位于偏移 num_authors 后的位置。
        self.emb = nn.Embedding(num_authors + num_papers, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def encode(self, adjacency: torch.Tensor) -> torch.Tensor:
        """执行多层邻居传播，并平均各层节点表示。

        LightGCN 不使用线性层和激活函数，只保留推荐系统中最核心的
        邻居信息聚合。最终表示是第 0 层初始嵌入与各传播层嵌入的均值。
        """
        layer_embeddings = [self.emb.weight]
        hidden = self.emb.weight
        for _ in range(self.num_layers):
            # 稀疏邻接矩阵乘节点表示，相当于聚合一跳邻居的信息。
            hidden = torch.sparse.mm(adjacency, hidden)
            layer_embeddings.append(hidden)
        return torch.stack(layer_embeddings, dim=0).mean(dim=0)

    def score(
        self,
        node_embeddings: torch.Tensor,
        author_ids: torch.Tensor,
        paper_ids: torch.Tensor,
    ) -> torch.Tensor:
        """通过作者向量与论文向量的点积计算匹配 logit。"""
        author_vectors = node_embeddings[author_ids]
        paper_vectors = node_embeddings[paper_ids + self.num_authors]
        return (author_vectors * paper_vectors).sum(dim=-1)


class HeteroGNNLayer(nn.Module):
    """一层同时更新作者节点和论文节点的异构消息传播模块。"""

    def __init__(
        self,
        author_in_dim: int,
        paper_in_dim: int,
        out_dim: int,
        dropout: float = 0.2,
    ) -> None:
        """为自环消息和不同关系消息创建独立的线性变换。"""
        super().__init__()

        # 每类节点保留一条自身消息，并接收来自另外两种关系的消息。
        # 不同关系使用独立参数，避免把合作、引用和作者-论文关系混为一谈。
        self.author_self = nn.Linear(author_in_dim, out_dim)
        self.paper_self = nn.Linear(paper_in_dim, out_dim)
        self.paper_to_author = nn.Linear(
            paper_in_dim,
            out_dim,
            bias=False,
        )
        self.author_author = nn.Linear(
            author_in_dim,
            out_dim,
            bias=False,
        )
        self.author_to_paper = nn.Linear(
            author_in_dim,
            out_dim,
            bias=False,
        )
        self.paper_paper = nn.Linear(
            paper_in_dim,
            out_dim,
            bias=False,
        )

        # 三个可训练标量经 softmax 后成为关系权重，权重之和为 1。
        self.author_rel_weight = nn.Parameter(torch.ones(3))
        self.paper_rel_weight = nn.Parameter(torch.ones(3))
        self.author_norm = nn.LayerNorm(out_dim)
        self.paper_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        author_hidden: torch.Tensor,
        paper_hidden: torch.Tensor,
        adjacencies: RelationAdjacencies,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """分别聚合作者侧和论文侧的关系消息。

        输入形状为 ``[节点数, 输入维度]``，输出形状为
        ``[节点数, out_dim]``。稀疏矩阵的行是目标节点，列是源节点。
        """
        # 作者从关联论文和合作作者处接收信息。
        author_from_paper = torch.sparse.mm(
            adjacencies["paper_to_author"],
            paper_hidden,
        )
        author_from_author = torch.sparse.mm(
            adjacencies["author_author"],
            author_hidden,
        )
        # 论文从关联作者和引用邻居论文处接收信息。
        paper_from_author = torch.sparse.mm(
            adjacencies["author_to_paper"],
            author_hidden,
        )
        paper_from_paper = torch.sparse.mm(
            adjacencies["paper_paper"],
            paper_hidden,
        )

        # 将自身消息与两类邻居消息堆叠，第一维固定为三种消息来源。
        author_messages = torch.stack(
            [
                self.author_self(author_hidden),
                self.paper_to_author(author_from_paper),
                self.author_author(author_from_author),
            ],
            dim=0,
        )
        paper_messages = torch.stack(
            [
                self.paper_self(paper_hidden),
                self.author_to_paper(paper_from_author),
                self.paper_paper(paper_from_paper),
            ],
            dim=0,
        )

        # softmax 让模型自动学习三种来源在当前节点类型中的相对重要性。
        author_weights = torch.softmax(
            self.author_rel_weight,
            dim=0,
        ).view(3, 1, 1)
        paper_weights = torch.softmax(
            self.paper_rel_weight,
            dim=0,
        ).view(3, 1, 1)
        author_output = (author_messages * author_weights).sum(dim=0)
        paper_output = (paper_messages * paper_weights).sum(dim=0)

        # 输入输出维度一致时使用残差连接，缓解深层传播造成的信息丢失。
        author_residual = (
            author_hidden if author_hidden.size(1) == author_output.size(1) else None
        )
        paper_residual = (
            paper_hidden if paper_hidden.size(1) == paper_output.size(1) else None
        )

        # LayerNorm 稳定特征尺度，ReLU 引入非线性，Dropout 抑制过拟合。
        author_output = self.dropout(torch.relu(self.author_norm(author_output)))
        paper_output = self.dropout(torch.relu(self.paper_norm(paper_output)))
        if author_residual is not None:
            author_output = author_output + author_residual
        if paper_residual is not None:
            paper_output = paper_output + paper_residual
        return author_output, paper_output


class HeteroGNN(nn.Module):
    """融合论文内容特征和三类图关系的异构图神经网络。"""

    def __init__(
        self,
        num_authors: int,
        paper_features: torch.Tensor,
        author_features: torch.Tensor,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """初始化节点输入、消息传播层和作者-论文解码器。"""
        super().__init__()
        paper_features = paper_features.float()
        author_features = author_features.float()
        paper_feature_dim = paper_features.size(1)

        # 作者基础特征是历史论文特征的平均值；可训练嵌入用于补充平均
        # 特征无法表达的作者个性、活跃度等隐含信息。
        self.author_emb = nn.Embedding(
            num_authors,
            paper_feature_dim,
        )
        self.input_author_norm = nn.LayerNorm(paper_feature_dim)
        self.input_paper_norm = nn.LayerNorm(paper_feature_dim)
        self.input_dropout = nn.Dropout(dropout)
        # register_buffer 会让特征随模型移动到 GPU，但不会被优化器更新。
        self.register_buffer(
            "author_features",
            author_features,
            persistent=False,
        )
        self.register_buffer(
            "paper_features",
            paper_features,
            persistent=False,
        )

        message_passing_layers = []
        author_dim = paper_feature_dim
        paper_dim = paper_feature_dim
        for layer_index in range(num_layers):
            # 中间层使用 hidden_dim，最后一层输出统一的 out_dim。
            next_dim = out_dim if layer_index == num_layers - 1 else hidden_dim
            message_passing_layers.append(
                HeteroGNNLayer(
                    author_dim,
                    paper_dim,
                    next_dim,
                    dropout=dropout,
                )
            )
            author_dim = next_dim
            paper_dim = next_dim
        self.layers = nn.ModuleList(message_passing_layers)

        # 解码器同时观察作者向量、论文向量、逐元素乘积和绝对差值。
        # 这四种视角分别保留节点身份、相互作用和距离信息。
        self.decoder = nn.Sequential(
            nn.Linear(out_dim * 4, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )
        self.author_bias = nn.Embedding(num_authors, 1)
        self.paper_bias = nn.Embedding(paper_features.size(0), 1)
        # 两个可训练缩放系数控制嵌入点积和原始语义相似度的贡献。
        self.dot_scale = nn.Parameter(torch.tensor(1.0))
        self.feature_scale = nn.Parameter(torch.tensor(1.0))

        nn.init.xavier_uniform_(self.author_emb.weight)
        nn.init.zeros_(self.author_bias.weight)
        nn.init.zeros_(self.paper_bias.weight)

    def encode(
        self,
        adjacencies: RelationAdjacencies,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构造初始节点表示，并经过多层消息传播生成最终嵌入。"""
        # 作者输入 = 历史论文平均特征 + 作者专属可训练嵌入。
        author_hidden = self.input_dropout(
            torch.relu(
                self.input_author_norm(self.author_features + self.author_emb.weight)
            )
        )
        # 论文节点直接使用 feature.pkl 中的 512 维内容特征。
        paper_hidden = self.input_dropout(
            torch.relu(self.input_paper_norm(self.paper_features))
        )

        # 每一层都会综合自身、跨类型邻居和同类型邻居的信息。
        for layer in self.layers:
            author_hidden, paper_hidden = layer(
                author_hidden,
                paper_hidden,
                adjacencies,
            )
        return author_hidden, paper_hidden

    def score(
        self,
        author_embeddings: torch.Tensor,
        paper_embeddings: torch.Tensor,
        author_ids: torch.Tensor,
        paper_ids: torch.Tensor,
    ) -> torch.Tensor:
        """融合图嵌入、原始语义相似度与节点偏置得到最终 logit。"""
        author_vectors = author_embeddings[author_ids]
        paper_vectors = paper_embeddings[paper_ids]
        # MLP 输入维度为 out_dim * 4，对每个候选 pair 独立打分。
        pair_features = torch.cat(
            [
                author_vectors,
                paper_vectors,
                author_vectors * paper_vectors,
                torch.abs(author_vectors - paper_vectors),
            ],
            dim=-1,
        )

        # MLP 可以学习比简单点积更复杂的非线性交互模式。
        mlp_score = self.decoder(pair_features).squeeze(-1)
        # 点积衡量 GNN 嵌入空间中的方向一致性。
        dot_score = (author_vectors * paper_vectors).sum(dim=-1)
        # 原始特征余弦相似度保留未经 GNN 传播的内容匹配证据。
        feature_score = (
            F.normalize(self.author_features[author_ids], dim=-1)
            * F.normalize(self.paper_features[paper_ids], dim=-1)
        ).sum(dim=-1)
        bias = self.author_bias(author_ids).squeeze(-1) + self.paper_bias(
            paper_ids
        ).squeeze(-1)
        # 返回未经过 sigmoid 的 logit，损失函数内部会完成概率变换。
        return (
            mlp_score
            + self.dot_scale * dot_score
            + self.feature_scale * feature_score
            + bias
        )


class HeteroGNNCalibrator:
    """将神经网络分数和可解释结构特征融合为最终概率。"""

    def __init__(
        self,
        seed: int = 0,
        learning_rate: float = 0.025,
        max_iter: int = 600,
        max_leaf_nodes: int = 63,
        min_samples_leaf: int = 40,
        l2_regularization: float = 4.0,
        n_iter_no_change: int = 10,
    ) -> None:
        """初始化直方图梯度提升分类器。

        校准器位于 GNN 之后，负责学习不同证据之间的非线性组合，而
        不是替代 GNN 的节点表示学习。
        """
        self.model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            n_iter_no_change=n_iter_no_change,
            random_state=seed,
        )

    @staticmethod
    def _combine(
        model_scores: np.ndarray,
        structural_features: np.ndarray,
    ) -> np.ndarray:
        """把模型概率转换为 logit，再与结构特征按列拼接。

        概率先裁剪到安全范围，避免计算 ``log(0)`` 导致无穷大。
        """
        epsilon = 1e-6
        model_scores = np.asarray(model_scores)
        if model_scores.ndim == 1:
            # 单模型分数统一转换为二维列矩阵，兼容快照集成输入。
            model_scores = model_scores[:, None]

        clipped_scores = np.clip(
            model_scores,
            epsilon,
            1.0 - epsilon,
        )
        model_logits = np.log(clipped_scores / (1.0 - clipped_scores))
        return np.column_stack([model_logits, structural_features])

    def fit(
        self,
        model_scores: np.ndarray,
        structural_features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """使用神经分数、结构特征和标签训练校准器。"""
        features = self._combine(model_scores, structural_features)
        self.model.fit(features, labels)

    def predict_proba(
        self,
        model_scores: np.ndarray,
        structural_features: np.ndarray,
    ) -> np.ndarray:
        """返回校准后属于正样本类别的概率。"""
        features = self._combine(model_scores, structural_features)
        return self.model.predict_proba(features)[:, 1]

    def explain(
        self,
        model_scores: np.ndarray,
        structural_features: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
        model_names: Optional[list[str]] = None,
    ) -> list[tuple[str, float]]:
        """通过置换重要性解释各输入对 ROC AUC 的贡献。

        逐列随机打乱特征并观察 AUC 下降幅度；下降越多，说明该特征
        对当前校准器越重要。
        """
        features = self._combine(model_scores, structural_features)
        result = permutation_importance(
            self.model,
            features,
            labels,
            scoring="roc_auc",
            n_repeats=3,
            random_state=0,
        )

        score_count = 1 if model_scores.ndim == 1 else model_scores.shape[1]
        # 为神经模型列生成可读名称，方便和 89 个结构特征共同排序。
        if score_count == 1:
            score_names = ["heterognn_logit"]
        elif model_names is not None and len(model_names) == score_count:
            score_names = [f"{name}_logit" for name in model_names]
        else:
            score_names = [f"model_logit_{index}" for index in range(score_count)]

        names = score_names + feature_names
        importance = zip(names, result.importances_mean)
        return sorted(
            importance,
            key=lambda item: item[1],
            reverse=True,
        )
