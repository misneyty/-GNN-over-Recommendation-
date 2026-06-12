from typing import Optional, Union

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class StructuralFeatureStore:
    feature_names = [
        "coauthor_support_log",
        "citation_forward_support_log",
        "citation_reverse_support_log",
        "second_coauthor_support_log",
        "coauthor_citation_forward_log",
        "coauthor_citation_reverse_log",
        "has_coauthor_support",
        "has_citation_forward_support",
        "has_citation_reverse_support",
        "has_second_coauthor_support",
        "has_coauthor_citation_forward",
        "has_coauthor_citation_reverse",
        "author_paper_degree_log",
        "paper_author_degree_log",
        "author_coauthor_degree_log",
        "paper_citation_out_degree_log",
        "paper_citation_in_degree_log",
        "coauthor_coverage",
        "candidate_author_coverage",
        "coauthor_jaccard",
        "citation_forward_history_coverage",
        "citation_forward_candidate_coverage",
        "citation_reverse_history_coverage",
        "citation_reverse_candidate_coverage",
        "specific_coauthor_support",
        "specific_citation_forward_support",
        "specific_citation_reverse_support",
        "coauthor_transition_probability",
        "second_coauthor_transition_probability",
        "citation_two_hop_forward_probability",
        "citation_two_hop_reverse_probability",
        "citation_two_hop_forward_log",
        "citation_two_hop_reverse_log",
        "citation_common_out_neighbor_log",
        "citation_common_in_neighbor_log",
        "citation_common_out_unique_log",
        "citation_common_in_unique_log",
        "citation_common_out_jaccard",
        "citation_common_in_jaccard",
        "citation_common_out_cosine",
        "citation_common_in_cosine",
        "specific_citation_common_out",
        "specific_citation_common_in",
        "has_citation_two_hop_forward",
        "has_citation_two_hop_reverse",
        "has_citation_common_out_neighbor",
        "has_citation_common_in_neighbor",
        "citation_two_hop_forward_history_coverage",
        "citation_two_hop_reverse_history_coverage",
        "citation_common_out_history_coverage",
        "citation_common_in_history_coverage",
        "collaborative_support_log",
        "collaborative_neighbor_count_log",
        "has_collaborative_support",
        "collaborative_candidate_coverage",
        "collaborative_author_coverage",
        "collaborative_mean_similarity",
        "semantic_collaborative_support_log",
        "semantic_collaborative_neighbor_count_log",
        "has_semantic_collaborative_support",
        "semantic_collaborative_candidate_coverage",
        "semantic_collaborative_author_coverage",
        "semantic_collaborative_mean_similarity",
        "candidate_team_similarity_maximum",
        "candidate_team_similarity_top_three_mean",
        "candidate_team_similarity_mean",
        "candidate_team_similarity_sum_log",
        "candidate_team_similar_author_count_log",
        "candidate_team_similar_author_fraction",
        "candidate_team_semantic_similarity_maximum",
        "candidate_team_semantic_similarity_top_three_mean",
        "candidate_team_semantic_similarity_mean",
        "semantic_author_profile_similarity",
        "semantic_author_to_candidate_reference_profile",
        "semantic_author_to_candidate_citing_profile",
        "semantic_author_reference_to_candidate",
        "semantic_author_citing_to_candidate",
        "semantic_reference_profile_alignment",
        "semantic_citing_profile_alignment",
        "semantic_maximum",
        "semantic_top_three_mean",
        "semantic_top_five_mean",
        "semantic_history_mean",
        "semantic_history_std",
        "semantic_history_median",
        "semantic_high_similarity_fraction",
        "semantic_maximum_margin",
        "semantic_top_three_margin",
        "semantic_maximum_zscore",
    ]

    def __init__(
        self,
        num_authors: int,
        num_papers: int,
        train_edges: np.ndarray,
        coauthor_edges: np.ndarray,
        citation_edges: np.ndarray,
        paper_features: np.ndarray,
        collaborative_neighbors: int = 30,
    ) -> None:
        self.paper_features = paper_features
        self.author_paper = sparse.csr_matrix(
            (
                np.ones(len(train_edges), dtype=np.float32),
                (train_edges[:, 0], train_edges[:, 1]),
            ),
            shape=(num_authors, num_papers),
        )
        author_author = sparse.csr_matrix(
            (
                np.ones(len(coauthor_edges) * 2, dtype=np.float32),
                (
                    np.concatenate([coauthor_edges[:, 0], coauthor_edges[:, 1]]),
                    np.concatenate([coauthor_edges[:, 1], coauthor_edges[:, 0]]),
                ),
            ),
            shape=(num_authors, num_authors),
        )
        paper_paper_forward = sparse.csr_matrix(
            (
                np.ones(len(citation_edges), dtype=np.float32),
                (citation_edges[:, 0], citation_edges[:, 1]),
            ),
            shape=(num_papers, num_papers),
        )

        self.author_degree = np.asarray(self.author_paper.sum(axis=1)).ravel()
        self.paper_degree = np.asarray(self.author_paper.sum(axis=0)).ravel()
        self.paper_author = self.author_paper.T.tocsr()
        self.coauthor_degree = np.asarray(author_author.sum(axis=1)).ravel()
        self.paper_citation_out_degree = np.asarray(
            paper_paper_forward.sum(axis=1)
        ).ravel()
        self.paper_citation_in_degree = np.asarray(
            paper_paper_forward.sum(axis=0)
        ).ravel()
        author_paper_transition = self._row_normalize(self.author_paper)
        author_author_transition = self._row_normalize(author_author)
        paper_forward_transition = self._row_normalize(paper_paper_forward)
        paper_reverse_transition = self._row_normalize(
            paper_paper_forward.T.tocsr()
        )
        author_specificity = 1.0 / np.log2(2.0 + self.author_degree)
        paper_connectivity = (
            self.paper_degree
            + self.paper_citation_out_degree
            + self.paper_citation_in_degree
        )
        history_paper_specificity = 1.0 / np.log2(2.0 + paper_connectivity)
        weighted_author_paper = self.author_paper.multiply(
            author_specificity[:, None]
        )
        weighted_history = self.author_paper.multiply(
            history_paper_specificity[None, :]
        )
        self.coauthor_support = (author_author @ self.author_paper).tocsr()
        self.specific_coauthor_support = (
            author_author @ weighted_author_paper
        ).tocsr()
        self.citation_forward_support = (
            self.author_paper @ paper_paper_forward
        ).tocsr()
        self.citation_reverse_support = (
            self.author_paper @ paper_paper_forward.T
        ).tocsr()
        self.specific_citation_forward_support = (
            weighted_history @ paper_paper_forward
        ).tocsr()
        self.specific_citation_reverse_support = (
            weighted_history @ paper_paper_forward.T
        ).tocsr()
        self.coauthor_transition_probability = (
            author_author_transition @ author_paper_transition
        ).tocsr()
        self.second_coauthor_transition_probability = (
            author_author_transition @ self.coauthor_transition_probability
        ).tocsr()
        self.citation_two_hop_forward_probability = (
            author_paper_transition
            @ paper_forward_transition
            @ paper_forward_transition
        ).tocsr()
        self.citation_two_hop_reverse_probability = (
            author_paper_transition
            @ paper_reverse_transition
            @ paper_reverse_transition
        ).tocsr()
        self.citation_two_hop_forward_support = (
            self.citation_forward_support @ paper_paper_forward
        ).tocsr()
        self.citation_two_hop_reverse_support = (
            self.citation_reverse_support @ paper_paper_forward.T
        ).tocsr()
        self.citation_common_out_neighbor_support = (
            self.citation_forward_support @ paper_paper_forward.T
        ).tocsr()
        self.citation_common_in_neighbor_support = (
            self.citation_reverse_support @ paper_paper_forward
        ).tocsr()
        author_out_neighbors = self.citation_forward_support.copy()
        author_out_neighbors.data = np.ones_like(
            author_out_neighbors.data,
            dtype=np.float32,
        )
        author_in_neighbors = self.citation_reverse_support.copy()
        author_in_neighbors.data = np.ones_like(
            author_in_neighbors.data,
            dtype=np.float32,
        )
        self.author_out_neighbor_degree = np.asarray(
            author_out_neighbors.sum(axis=1)
        ).ravel()
        self.author_in_neighbor_degree = np.asarray(
            author_in_neighbors.sum(axis=1)
        ).ravel()
        self.citation_common_out_unique = (
            author_out_neighbors @ paper_paper_forward.T
        ).tocsr()
        self.citation_common_in_unique = (
            author_in_neighbors @ paper_paper_forward
        ).tocsr()
        cited_paper_specificity = (
            1.0 / np.log2(2.0 + self.paper_citation_in_degree)
        )
        citing_paper_specificity = (
            1.0 / np.log2(2.0 + self.paper_citation_out_degree)
        )
        self.specific_citation_common_out = (
            author_out_neighbors.multiply(
                cited_paper_specificity[None, :]
            )
            @ paper_paper_forward.T
        ).tocsr()
        self.specific_citation_common_in = (
            author_in_neighbors.multiply(
                citing_paper_specificity[None, :]
            )
            @ paper_paper_forward
        ).tocsr()
        self.second_coauthor_support = (
            author_author @ self.coauthor_support
        ).tocsr()
        self.coauthor_citation_forward_support = (
            self.coauthor_support @ paper_paper_forward
        ).tocsr()
        self.coauthor_citation_reverse_support = (
            self.coauthor_support @ paper_paper_forward.T
        ).tocsr()
        self.collaborative_author_similarity = (
            self._build_collaborative_author_similarity(
                self.author_paper,
                collaborative_neighbors,
            )
        )
        collaborative_binary = self.collaborative_author_similarity.copy()
        collaborative_binary.data = np.ones_like(
            collaborative_binary.data,
            dtype=np.float32,
        )
        self.collaborative_similarity_mass = np.asarray(
            self.collaborative_author_similarity.sum(axis=1)
        ).ravel()
        self.collaborative_support = (
            self.collaborative_author_similarity @ self.author_paper
        ).tocsr()
        self.collaborative_neighbor_count = (
            collaborative_binary @ self.author_paper
        ).tocsr()
        self.author_history_similarity = cosine_similarity(
            self.author_paper,
            dense_output=True,
        ).astype(np.float32, copy=False)
        np.fill_diagonal(self.author_history_similarity, 0.0)
        author_semantic_profiles = (
            self.author_paper @ self.paper_features
        ).astype(np.float32, copy=False)
        author_semantic_profiles /= np.maximum(
            self.author_degree[:, None],
            1.0,
        )
        self.author_semantic_profiles = self._l2_normalize_dense(
            author_semantic_profiles
        )
        self.normalized_paper_features = self._l2_normalize_dense(
            self.paper_features
        )
        self.paper_reference_semantic_profiles = self._l2_normalize_dense(
            paper_paper_forward @ self.normalized_paper_features
        )
        self.paper_citing_semantic_profiles = self._l2_normalize_dense(
            paper_paper_forward.T @ self.normalized_paper_features
        )
        self.author_reference_semantic_profiles = self._l2_normalize_dense(
            self.citation_forward_support
            @ self.normalized_paper_features
        )
        self.author_citing_semantic_profiles = self._l2_normalize_dense(
            self.citation_reverse_support
            @ self.normalized_paper_features
        )
        self.semantic_author_similarity = (
            self._build_collaborative_author_similarity(
                self.author_semantic_profiles,
                collaborative_neighbors,
            )
        )
        semantic_collaborative_binary = self.semantic_author_similarity.copy()
        semantic_collaborative_binary.data = np.ones_like(
            semantic_collaborative_binary.data,
            dtype=np.float32,
        )
        self.semantic_collaborative_similarity_mass = np.asarray(
            self.semantic_author_similarity.sum(axis=1)
        ).ravel()
        self.semantic_collaborative_support = (
            self.semantic_author_similarity @ self.author_paper
        ).tocsr()
        self.semantic_collaborative_neighbor_count = (
            semantic_collaborative_binary @ self.author_paper
        ).tocsr()

    @staticmethod
    def _row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        row_sum = np.asarray(matrix.sum(axis=1)).ravel()
        inverse = np.zeros_like(row_sum, dtype=np.float32)
        nonzero = row_sum > 0
        inverse[nonzero] = 1.0 / row_sum[nonzero]
        return matrix.multiply(inverse[:, None]).tocsr()

    @staticmethod
    def _l2_normalize_dense(matrix: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norm, 1e-12)

    @staticmethod
    def _build_collaborative_author_similarity(
        author_representation: Union[sparse.csr_matrix, np.ndarray],
        neighbor_count: int,
    ) -> sparse.csr_matrix:
        num_authors = author_representation.shape[0]
        search_count = min(neighbor_count + 1, num_authors)
        nearest_neighbors = NearestNeighbors(
            n_neighbors=search_count,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nearest_neighbors.fit(author_representation)
        distances, neighbor_ids = nearest_neighbors.kneighbors(
            author_representation
        )

        row_ids = np.repeat(np.arange(num_authors), search_count)
        column_ids = neighbor_ids.ravel()
        similarities = (1.0 - distances).ravel().astype(np.float32)
        keep = (row_ids != column_ids) & (similarities > 0.0)
        return sparse.csr_matrix(
            (
                similarities[keep],
                (row_ids[keep], column_ids[keep]),
            ),
            shape=(num_authors, num_authors),
        )

    @staticmethod
    def _gather(matrix: sparse.csr_matrix, edges: np.ndarray) -> np.ndarray:
        return np.asarray(matrix[edges[:, 0], edges[:, 1]]).ravel()

    def _semantic_history_features(
        self,
        edges: np.ndarray,
        device: Optional[torch.device],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        device = device or torch.device("cpu")
        normalized_features = torch.nn.functional.normalize(
            torch.as_tensor(self.paper_features, dtype=torch.float32, device=device),
            dim=1,
        )
        maximum = np.zeros(len(edges), dtype=np.float32)
        top_three_mean = np.zeros(len(edges), dtype=np.float32)
        top_five_mean = np.zeros(len(edges), dtype=np.float32)
        history_mean = np.zeros(len(edges), dtype=np.float32)
        history_std = np.zeros(len(edges), dtype=np.float32)
        history_median = np.zeros(len(edges), dtype=np.float32)
        high_similarity_fraction = np.zeros(len(edges), dtype=np.float32)
        edge_order = np.argsort(edges[:, 0], kind="stable")
        sorted_authors = edges[edge_order, 0]
        boundaries = np.flatnonzero(np.diff(sorted_authors)) + 1

        with torch.no_grad():
            for indices in np.split(edge_order, boundaries):
                author_id = int(edges[indices[0], 0])
                start = self.author_paper.indptr[author_id]
                end = self.author_paper.indptr[author_id + 1]
                history_ids = self.author_paper.indices[start:end]
                if len(history_ids) == 0:
                    continue

                candidate_ids = edges[indices, 1]
                candidate_tensor = torch.as_tensor(
                    candidate_ids,
                    dtype=torch.long,
                    device=device,
                )
                history_tensor = torch.as_tensor(
                    history_ids,
                    dtype=torch.long,
                    device=device,
                )
                similarities = (
                    normalized_features[candidate_tensor]
                    @ normalized_features[history_tensor].T
                )
                top_three_count = min(3, similarities.size(1))
                top_three_values = torch.topk(
                    similarities,
                    k=top_three_count,
                    dim=1,
                ).values
                top_five_count = min(5, similarities.size(1))
                top_five_values = torch.topk(
                    similarities,
                    k=top_five_count,
                    dim=1,
                ).values
                maximum[indices] = top_three_values[:, 0].cpu().numpy()
                top_three_mean[indices] = (
                    top_three_values.mean(dim=1).cpu().numpy()
                )
                top_five_mean[indices] = (
                    top_five_values.mean(dim=1).cpu().numpy()
                )
                history_mean[indices] = similarities.mean(dim=1).cpu().numpy()
                history_std[indices] = similarities.std(
                    dim=1,
                    unbiased=False,
                ).cpu().numpy()
                similarity_values = similarities.cpu().numpy()
                median_index = (similarity_values.shape[1] - 1) // 2
                history_median[indices] = np.partition(
                    similarity_values,
                    median_index,
                    axis=1,
                )[:, median_index]
                high_similarity_fraction[indices] = (
                    similarities >= 0.5
                ).float().mean(dim=1).cpu().numpy()
        return (
            maximum,
            top_three_mean,
            top_five_mean,
            history_mean,
            history_std,
            history_median,
            high_similarity_fraction,
        )

    def _candidate_team_similarity_features(
        self,
        edges: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        maximum = np.zeros(len(edges), dtype=np.float32)
        top_three_mean = np.zeros(len(edges), dtype=np.float32)
        mean = np.zeros(len(edges), dtype=np.float32)
        total = np.zeros(len(edges), dtype=np.float32)
        similar_author_count = np.zeros(len(edges), dtype=np.float32)
        similar_author_fraction = np.zeros(len(edges), dtype=np.float32)
        edge_order = np.argsort(edges[:, 1], kind="stable")
        sorted_papers = edges[edge_order, 1]
        boundaries = np.flatnonzero(np.diff(sorted_papers)) + 1

        for indices in np.split(edge_order, boundaries):
            paper_id = int(edges[indices[0], 1])
            start = self.paper_author.indptr[paper_id]
            end = self.paper_author.indptr[paper_id + 1]
            team_ids = self.paper_author.indices[start:end]
            if len(team_ids) == 0:
                continue

            author_ids = edges[indices, 0]
            similarities = self.author_history_similarity[
                author_ids[:, None],
                team_ids[None, :],
            ]
            team_sizes = len(team_ids) - np.isin(author_ids, team_ids).astype(
                np.int64
            )
            team_sizes = np.maximum(team_sizes, 1)
            top_count = min(3, similarities.shape[1])
            top_values = np.partition(
                similarities,
                similarities.shape[1] - top_count,
                axis=1,
            )[:, -top_count:]
            row_total = similarities.sum(axis=1)
            row_similar_count = (similarities > 0.0).sum(axis=1)

            maximum[indices] = top_values.max(axis=1)
            top_three_mean[indices] = top_values.sum(axis=1) / np.minimum(
                top_count,
                team_sizes,
            )
            mean[indices] = row_total / team_sizes
            total[indices] = row_total
            similar_author_count[indices] = row_similar_count
            similar_author_fraction[indices] = row_similar_count / team_sizes

        return (
            maximum,
            top_three_mean,
            mean,
            total,
            similar_author_count,
            similar_author_fraction,
        )

    def _candidate_team_semantic_similarity_features(
        self,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        maximum = np.zeros(len(edges), dtype=np.float32)
        top_three_mean = np.zeros(len(edges), dtype=np.float32)
        mean = np.zeros(len(edges), dtype=np.float32)
        edge_order = np.argsort(edges[:, 1], kind="stable")
        sorted_papers = edges[edge_order, 1]
        boundaries = np.flatnonzero(np.diff(sorted_papers)) + 1

        for indices in np.split(edge_order, boundaries):
            paper_id = int(edges[indices[0], 1])
            start = self.paper_author.indptr[paper_id]
            end = self.paper_author.indptr[paper_id + 1]
            team_ids = self.paper_author.indices[start:end]
            if len(team_ids) == 0:
                continue

            author_ids = edges[indices, 0]
            similarities = (
                self.author_semantic_profiles[author_ids]
                @ self.author_semantic_profiles[team_ids].T
            )
            same_author = author_ids[:, None] == team_ids[None, :]
            similarities[same_author] = 0.0
            team_sizes = len(team_ids) - same_author.sum(axis=1)
            team_sizes = np.maximum(team_sizes, 1)
            top_count = min(3, similarities.shape[1])
            top_values = np.partition(
                similarities,
                similarities.shape[1] - top_count,
                axis=1,
            )[:, -top_count:]

            maximum[indices] = top_values.max(axis=1)
            top_three_mean[indices] = top_values.sum(axis=1) / np.minimum(
                top_count,
                team_sizes,
            )
            mean[indices] = similarities.sum(axis=1) / team_sizes

        return maximum, top_three_mean, mean

    def transform(
        self,
        edges: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        coauthor = self._gather(self.coauthor_support, edges)
        citation_forward = self._gather(self.citation_forward_support, edges)
        citation_reverse = self._gather(self.citation_reverse_support, edges)
        specific_coauthor = self._gather(
            self.specific_coauthor_support,
            edges,
        )
        specific_citation_forward = self._gather(
            self.specific_citation_forward_support,
            edges,
        )
        specific_citation_reverse = self._gather(
            self.specific_citation_reverse_support,
            edges,
        )
        coauthor_transition_probability = self._gather(
            self.coauthor_transition_probability,
            edges,
        )
        second_coauthor_transition_probability = self._gather(
            self.second_coauthor_transition_probability,
            edges,
        )
        citation_two_hop_forward_probability = self._gather(
            self.citation_two_hop_forward_probability,
            edges,
        )
        citation_two_hop_reverse_probability = self._gather(
            self.citation_two_hop_reverse_probability,
            edges,
        )
        citation_two_hop_forward = self._gather(
            self.citation_two_hop_forward_support,
            edges,
        )
        citation_two_hop_reverse = self._gather(
            self.citation_two_hop_reverse_support,
            edges,
        )
        citation_common_out_neighbor = self._gather(
            self.citation_common_out_neighbor_support,
            edges,
        )
        citation_common_in_neighbor = self._gather(
            self.citation_common_in_neighbor_support,
            edges,
        )
        citation_common_out_unique = self._gather(
            self.citation_common_out_unique,
            edges,
        )
        citation_common_in_unique = self._gather(
            self.citation_common_in_unique,
            edges,
        )
        specific_citation_common_out = self._gather(
            self.specific_citation_common_out,
            edges,
        )
        specific_citation_common_in = self._gather(
            self.specific_citation_common_in,
            edges,
        )
        collaborative_support = self._gather(
            self.collaborative_support,
            edges,
        )
        collaborative_neighbor_count = self._gather(
            self.collaborative_neighbor_count,
            edges,
        )
        semantic_collaborative_support = self._gather(
            self.semantic_collaborative_support,
            edges,
        )
        semantic_collaborative_neighbor_count = self._gather(
            self.semantic_collaborative_neighbor_count,
            edges,
        )
        second_coauthor = self._gather(self.second_coauthor_support, edges)
        coauthor_citation_forward = self._gather(
            self.coauthor_citation_forward_support,
            edges,
        )
        coauthor_citation_reverse = self._gather(
            self.coauthor_citation_reverse_support,
            edges,
        )
        (
            candidate_team_similarity_maximum,
            candidate_team_similarity_top_three_mean,
            candidate_team_similarity_mean,
            candidate_team_similarity_sum,
            candidate_team_similar_author_count,
            candidate_team_similar_author_fraction,
        ) = self._candidate_team_similarity_features(edges)
        (
            candidate_team_semantic_similarity_maximum,
            candidate_team_semantic_similarity_top_three_mean,
            candidate_team_semantic_similarity_mean,
        ) = self._candidate_team_semantic_similarity_features(edges)
        (
            semantic_maximum,
            semantic_top_three_mean,
            semantic_top_five_mean,
            semantic_history_mean,
            semantic_history_std,
            semantic_history_median,
            semantic_high_similarity_fraction,
        ) = self._semantic_history_features(edges, device)
        author_ids = edges[:, 0]
        paper_ids = edges[:, 1]
        author_degree = np.maximum(self.author_degree[author_ids], 1.0)
        paper_degree = np.maximum(self.paper_degree[paper_ids], 1.0)
        coauthor_degree = np.maximum(self.coauthor_degree[author_ids], 1.0)
        citation_out_degree = np.maximum(
            self.paper_citation_out_degree[paper_ids],
            1.0,
        )
        citation_in_degree = np.maximum(
            self.paper_citation_in_degree[paper_ids],
            1.0,
        )
        coauthor_union = np.maximum(
            coauthor_degree + paper_degree - coauthor,
            1.0,
        )
        collaborative_mass = np.maximum(
            self.collaborative_similarity_mass[author_ids],
            1e-6,
        )
        collaborative_count_denominator = np.maximum(
            collaborative_neighbor_count,
            1.0,
        )
        semantic_collaborative_mass = np.maximum(
            self.semantic_collaborative_similarity_mass[author_ids],
            1e-6,
        )
        semantic_collaborative_count_denominator = np.maximum(
            semantic_collaborative_neighbor_count,
            1.0,
        )
        common_out_union = np.maximum(
            self.author_out_neighbor_degree[author_ids]
            + self.paper_citation_out_degree[paper_ids]
            - citation_common_out_unique,
            1.0,
        )
        common_in_union = np.maximum(
            self.author_in_neighbor_degree[author_ids]
            + self.paper_citation_in_degree[paper_ids]
            - citation_common_in_unique,
            1.0,
        )
        common_out_norm = np.sqrt(
            np.maximum(
                self.author_out_neighbor_degree[author_ids]
                * self.paper_citation_out_degree[paper_ids],
                1.0,
            )
        )
        common_in_norm = np.sqrt(
            np.maximum(
                self.author_in_neighbor_degree[author_ids]
                * self.paper_citation_in_degree[paper_ids],
                1.0,
            )
        )
        semantic_author_profile_similarity = np.sum(
            self.author_semantic_profiles[author_ids]
            * self.normalized_paper_features[paper_ids],
            axis=1,
        )
        semantic_author_to_candidate_reference_profile = np.sum(
            self.author_semantic_profiles[author_ids]
            * self.paper_reference_semantic_profiles[paper_ids],
            axis=1,
        )
        semantic_author_to_candidate_citing_profile = np.sum(
            self.author_semantic_profiles[author_ids]
            * self.paper_citing_semantic_profiles[paper_ids],
            axis=1,
        )
        semantic_author_reference_to_candidate = np.sum(
            self.author_reference_semantic_profiles[author_ids]
            * self.normalized_paper_features[paper_ids],
            axis=1,
        )
        semantic_author_citing_to_candidate = np.sum(
            self.author_citing_semantic_profiles[author_ids]
            * self.normalized_paper_features[paper_ids],
            axis=1,
        )
        semantic_reference_profile_alignment = np.sum(
            self.author_reference_semantic_profiles[author_ids]
            * self.paper_reference_semantic_profiles[paper_ids],
            axis=1,
        )
        semantic_citing_profile_alignment = np.sum(
            self.author_citing_semantic_profiles[author_ids]
            * self.paper_citing_semantic_profiles[paper_ids],
            axis=1,
        )
        semantic_standard_deviation = np.maximum(
            semantic_history_std,
            1e-4,
        )
        return np.column_stack(
            [
                np.log1p(coauthor),
                np.log1p(citation_forward),
                np.log1p(citation_reverse),
                np.log1p(second_coauthor),
                np.log1p(coauthor_citation_forward),
                np.log1p(coauthor_citation_reverse),
                coauthor > 0,
                citation_forward > 0,
                citation_reverse > 0,
                second_coauthor > 0,
                coauthor_citation_forward > 0,
                coauthor_citation_reverse > 0,
                np.log1p(self.author_degree[author_ids]),
                np.log1p(self.paper_degree[paper_ids]),
                np.log1p(self.coauthor_degree[author_ids]),
                np.log1p(self.paper_citation_out_degree[paper_ids]),
                np.log1p(self.paper_citation_in_degree[paper_ids]),
                coauthor / coauthor_degree,
                coauthor / paper_degree,
                coauthor / coauthor_union,
                citation_forward / author_degree,
                citation_forward / citation_in_degree,
                citation_reverse / author_degree,
                citation_reverse / citation_out_degree,
                specific_coauthor,
                specific_citation_forward,
                specific_citation_reverse,
                coauthor_transition_probability,
                second_coauthor_transition_probability,
                citation_two_hop_forward_probability,
                citation_two_hop_reverse_probability,
                np.log1p(citation_two_hop_forward),
                np.log1p(citation_two_hop_reverse),
                np.log1p(citation_common_out_neighbor),
                np.log1p(citation_common_in_neighbor),
                np.log1p(citation_common_out_unique),
                np.log1p(citation_common_in_unique),
                citation_common_out_unique / common_out_union,
                citation_common_in_unique / common_in_union,
                citation_common_out_unique / common_out_norm,
                citation_common_in_unique / common_in_norm,
                specific_citation_common_out,
                specific_citation_common_in,
                citation_two_hop_forward > 0,
                citation_two_hop_reverse > 0,
                citation_common_out_neighbor > 0,
                citation_common_in_neighbor > 0,
                citation_two_hop_forward / author_degree,
                citation_two_hop_reverse / author_degree,
                citation_common_out_neighbor / author_degree,
                citation_common_in_neighbor / author_degree,
                np.log1p(collaborative_support),
                np.log1p(collaborative_neighbor_count),
                collaborative_support > 0,
                collaborative_neighbor_count / paper_degree,
                collaborative_support / collaborative_mass,
                collaborative_support / collaborative_count_denominator,
                np.log1p(semantic_collaborative_support),
                np.log1p(semantic_collaborative_neighbor_count),
                semantic_collaborative_support > 0,
                semantic_collaborative_neighbor_count / paper_degree,
                (
                    semantic_collaborative_support
                    / semantic_collaborative_mass
                ),
                (
                    semantic_collaborative_support
                    / semantic_collaborative_count_denominator
                ),
                candidate_team_similarity_maximum,
                candidate_team_similarity_top_three_mean,
                candidate_team_similarity_mean,
                np.log1p(candidate_team_similarity_sum),
                np.log1p(candidate_team_similar_author_count),
                candidate_team_similar_author_fraction,
                candidate_team_semantic_similarity_maximum,
                candidate_team_semantic_similarity_top_three_mean,
                candidate_team_semantic_similarity_mean,
                semantic_author_profile_similarity,
                semantic_author_to_candidate_reference_profile,
                semantic_author_to_candidate_citing_profile,
                semantic_author_reference_to_candidate,
                semantic_author_citing_to_candidate,
                semantic_reference_profile_alignment,
                semantic_citing_profile_alignment,
                semantic_maximum,
                semantic_top_three_mean,
                semantic_top_five_mean,
                semantic_history_mean,
                semantic_history_std,
                semantic_history_median,
                semantic_high_similarity_fraction,
                semantic_maximum - semantic_history_mean,
                semantic_top_three_mean - semantic_history_mean,
                (
                    semantic_maximum - semantic_history_mean
                ) / semantic_standard_deviation,
            ]
        ).astype(np.float32)
