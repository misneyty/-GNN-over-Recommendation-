import os
import pickle
import numpy as np

class Dataset:
    def __init__(
        self,
        num_authors: int,
        num_papers: int,
        train_edges: np.ndarray,
        test_edges: np.ndarray,
        coauthor_edges: np.ndarray,
        citation_edges: np.ndarray,
        paper_features: np.ndarray
    ):
        self.num_authors = num_authors
        self.num_papers = num_papers
        self.train_edges = train_edges
        self.test_edges = test_edges
        self.coauthor_edges = coauthor_edges
        self.citation_edges = citation_edges
        self.paper_features = paper_features


def read_edge_list(path: str) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.int64)
    return data


def load_paper_features(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        features = pickle.load(f)
    return np.asarray(features, dtype=np.float32)


def load_dataset(data_dir: str = "data_file") -> Dataset:
    train_edges = read_edge_list(
        os.path.join(data_dir, "bipartite_train_ann.txt")
    )

    test_edges = read_edge_list(
        os.path.join(data_dir, "bipartite_test_ann.txt")
    )

    coauthor_edges = read_edge_list(
        os.path.join(data_dir, "author_file_ann.txt")
    )

    citation_edges = read_edge_list(
        os.path.join(data_dir, "paper_file_ann.txt")
    )

    paper_features = load_paper_features(
        os.path.join(data_dir, "feature.pkl")
    )

    max_author = max(
        int(train_edges[:, 0].max()),
        int(test_edges[:, 0].max()),
        int(coauthor_edges.max()),
    )

    max_paper = max(
        int(train_edges[:, 1].max()),
        int(test_edges[:, 1].max()),
        int(citation_edges.max()),
        paper_features.shape[0] - 1,
    )

    return Dataset(
        num_authors=max_author + 1,
        num_papers=max_paper + 1,
        train_edges=train_edges,
        test_edges=test_edges,
        coauthor_edges=coauthor_edges,
        citation_edges=citation_edges,
        paper_features=paper_features,
    )