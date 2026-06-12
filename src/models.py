import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

#lightGCN模型：更适合推荐系统的GCN
class LightGCN(nn.Module):
    def __init__(
        self,
        num_authors: int,
        num_papers: int,
        dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_authors = num_authors
        self.num_papers = num_papers
        self.num_layers = num_layers
        self.emb = nn.Embedding(num_authors + num_papers, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def encode(self, adj: torch.Tensor) -> torch.Tensor:
        embeddings = [self.emb.weight]#总embedding（num_authors+num_papers）
        h = self.emb.weight
        for _ in range(self.num_layers):
            h = torch.sparse.mm(adj, h)#稀疏矩阵乘H矩阵
            embeddings.append(h)
        return torch.stack(embeddings, dim=0).mean(dim=0)

    def score(
        self,
        z: torch.Tensor,
        author_ids: torch.Tensor,
        paper_ids: torch.Tensor,
    ) -> torch.Tensor:
        author_z = z[author_ids]
        paper_z = z[paper_ids + self.num_authors]
        return (author_z * paper_z).sum(dim=-1)


class HeteroGNNLayer(nn.Module):
    def __init__(self, author_in_dim: int, paper_in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.author_self = nn.Linear(author_in_dim, out_dim)
        self.paper_self = nn.Linear(paper_in_dim, out_dim)

        self.paper_to_author = nn.Linear(paper_in_dim, out_dim, bias=False)
        self.author_author = nn.Linear(author_in_dim, out_dim, bias=False)
        self.author_to_paper = nn.Linear(author_in_dim, out_dim, bias=False)
        self.paper_paper = nn.Linear(paper_in_dim, out_dim, bias=False)

        self.author_rel_weight = nn.Parameter(torch.ones(3))
        self.paper_rel_weight = nn.Parameter(torch.ones(3))
        self.author_norm = nn.LayerNorm(out_dim)
        self.paper_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        author_h: torch.Tensor,
        paper_h: torch.Tensor,
        adjs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        author_from_paper = torch.sparse.mm(adjs["paper_to_author"], paper_h)
        author_from_author = torch.sparse.mm(adjs["author_author"], author_h)
        paper_from_author = torch.sparse.mm(adjs["author_to_paper"], author_h)
        paper_from_paper = torch.sparse.mm(adjs["paper_paper"], paper_h)

        author_messages = torch.stack(
            [
                self.author_self(author_h),
                self.paper_to_author(author_from_paper),
                self.author_author(author_from_author),
            ],
            dim=0,
        )
        paper_messages = torch.stack(
            [
                self.paper_self(paper_h),
                self.author_to_paper(paper_from_author),
                self.paper_paper(paper_from_paper),
            ],
            dim=0,
        )

        author_weight = torch.softmax(self.author_rel_weight, dim=0).view(3, 1, 1)
        paper_weight = torch.softmax(self.paper_rel_weight, dim=0).view(3, 1, 1)
        author_out = (author_messages * author_weight).sum(dim=0)
        paper_out = (paper_messages * paper_weight).sum(dim=0)

        author_residual = author_h if author_h.size(1) == author_out.size(1) else None
        paper_residual = paper_h if paper_h.size(1) == paper_out.size(1) else None

        author_out = self.dropout(torch.relu(self.author_norm(author_out)))
        paper_out = self.dropout(torch.relu(self.paper_norm(paper_out)))
        if author_residual is not None:
            author_out = author_out + author_residual
        if paper_residual is not None:
            paper_out = paper_out + paper_residual
        return author_out, paper_out


class HeteroGNN(nn.Module):
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
        super().__init__()
        paper_features = paper_features.float()
        author_features = author_features.float()
        paper_feature_dim = paper_features.size(1)

        self.author_emb = nn.Embedding(num_authors, paper_feature_dim)
        self.input_author_norm = nn.LayerNorm(paper_feature_dim)
        self.input_paper_norm = nn.LayerNorm(paper_feature_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.register_buffer("author_features", author_features, persistent=False)
        self.register_buffer("paper_features", paper_features, persistent=False)

        layers = []
        author_dim = paper_feature_dim
        paper_dim = paper_feature_dim
        for layer_idx in range(num_layers):
            next_dim = out_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.append(HeteroGNNLayer(author_dim, paper_dim, next_dim, dropout=dropout))
            author_dim = next_dim
            paper_dim = next_dim
        self.layers = nn.ModuleList(layers)
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
        self.dot_scale = nn.Parameter(torch.tensor(1.0))
        self.feature_scale = nn.Parameter(torch.tensor(1.0))

        nn.init.xavier_uniform_(self.author_emb.weight)
        nn.init.zeros_(self.author_bias.weight)
        nn.init.zeros_(self.paper_bias.weight)

    def encode(self, adjs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        author_h = self.input_dropout(
            torch.relu(self.input_author_norm(self.author_features + self.author_emb.weight))
        )
        paper_h = self.input_dropout(torch.relu(self.input_paper_norm(self.paper_features)))
        for layer in self.layers:
            author_h, paper_h = layer(author_h, paper_h, adjs)
        return author_h, paper_h

    def score(
        self,
        author_z: torch.Tensor,
        paper_z: torch.Tensor,
        author_ids: torch.Tensor,
        paper_ids: torch.Tensor,
    ) -> torch.Tensor:
        author_vec = author_z[author_ids]
        paper_vec = paper_z[paper_ids]
        pair_features = torch.cat(
            [
                author_vec,
                paper_vec,
                author_vec * paper_vec,
                torch.abs(author_vec - paper_vec),
            ],
            dim=-1,
        )
        mlp_score = self.decoder(pair_features).squeeze(-1)
        dot_score = (author_vec * paper_vec).sum(dim=-1)
        feature_score = (
            F.normalize(self.author_features[author_ids], dim=-1)
            * F.normalize(self.paper_features[paper_ids], dim=-1)
        ).sum(dim=-1)
        bias = self.author_bias(author_ids).squeeze(-1) + self.paper_bias(paper_ids).squeeze(-1)
        return mlp_score + self.dot_scale * dot_score + self.feature_scale * feature_score + bias


class HeteroGNNCalibrator:
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
        epsilon = 1e-6
        model_scores = np.asarray(model_scores)
        if model_scores.ndim == 1:
            model_scores = model_scores[:, None]
        clipped = np.clip(model_scores, epsilon, 1.0 - epsilon)
        model_logits = np.log(clipped / (1.0 - clipped))
        return np.column_stack([model_logits, structural_features])

    def fit(
        self,
        model_scores: np.ndarray,
        structural_features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        features = self._combine(model_scores, structural_features)
        self.model.fit(features, labels)

    def predict_proba(
        self,
        model_scores: np.ndarray,
        structural_features: np.ndarray,
    ) -> np.ndarray:
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
        if score_count == 1:
            score_names = ["heterognn_logit"]
        elif model_names is not None and len(model_names) == score_count:
            score_names = [
                f"{name}_logit"
                for name in model_names
            ]
        else:
            score_names = [
                f"model_logit_{index}"
                for index in range(score_count)
            ]
        names = score_names + feature_names
        importance = zip(names, result.importances_mean)
        return sorted(importance, key=lambda item: item[1], reverse=True)
