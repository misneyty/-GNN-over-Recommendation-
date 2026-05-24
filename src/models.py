import torch
from torch import nn

#使用矩阵分解方法
class MatrixFactorization(nn.Module):#继承自Pytorch的神经网络基类
    def __init__(self, num_authors: int, num_papers: int, dim: int = 64) -> None:
        super().__init__()
        self.author_emb = nn.Embedding(num_authors, dim)#构建author的embedding表
        self.paper_emb = nn.Embedding(num_papers, dim)#构建paper的embedding表
        nn.init.xavier_uniform_(self.author_emb.weight)#初始化author表
        nn.init.xavier_uniform_(self.paper_emb.weight)#初始化paper表

    def forward(self, author_ids: torch.Tensor, paper_ids: torch.Tensor) -> torch.Tensor:
        author_z = self.author_emb(author_ids)
        paper_z = self.paper_emb(paper_ids)
        return (author_z * paper_z).sum(dim=-1)#无需归一化，后续不稳定可考虑加L2正则化

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


class HeteroLightGCN(nn.Module):
    def __init__(
        self,
        num_authors: int,
        num_papers: int,
        dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.author_emb = nn.Embedding(num_authors, dim)
        self.paper_emb = nn.Embedding(num_papers, dim)
        nn.init.xavier_uniform_(self.author_emb.weight)
        nn.init.xavier_uniform_(self.paper_emb.weight)

    def encode(self, adjs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        author_h = self.author_emb.weight
        paper_h = self.paper_emb.weight
        author_layers = [author_h]
        paper_layers = [paper_h]

        for _ in range(self.num_layers):
            author_messages = [
                torch.sparse.mm(adjs["paper_to_author"], paper_h),
                torch.sparse.mm(adjs["author_author"], author_h),
            ]
            paper_messages = [
                torch.sparse.mm(adjs["author_to_paper"], author_h),
                torch.sparse.mm(adjs["paper_paper"], paper_h),
            ]

            author_h = torch.stack(author_messages, dim=0).mean(dim=0)
            paper_h = torch.stack(paper_messages, dim=0).mean(dim=0)
            author_layers.append(author_h)
            paper_layers.append(paper_h)

        author_z = torch.stack(author_layers, dim=0).mean(dim=0)
        paper_z = torch.stack(paper_layers, dim=0).mean(dim=0)
        return author_z, paper_z

    def score(
        self,
        author_z: torch.Tensor,
        paper_z: torch.Tensor,
        author_ids: torch.Tensor,
        paper_ids: torch.Tensor,
    ) -> torch.Tensor:
        return (author_z[author_ids] * paper_z[paper_ids]).sum(dim=-1)


class PaperFeatureMLP(nn.Module):

    def __init__(self, author_dim: int, paper_feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(author_dim + paper_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, author_z: torch.Tensor, paper_x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([author_z, paper_x], dim=-1)).squeeze(-1)
