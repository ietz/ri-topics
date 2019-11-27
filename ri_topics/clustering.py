from dataclasses import dataclass
from typing import Optional

import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
import umap


@dataclass
class ClusterAssignment:
    labels: np.ndarray
    probabilities: np.ndarray


class Clusterer:
    """Clustering using UMAP and HDBSCAN"""
    def __init__(self):
        self.umap: Optional[umap.UMAP] = None
        self.hdbscan: Optional[hdbscan.HDBSCAN] = None

    def fit(self, embeddings: np.ndarray, n_components: int, n_neighbors: int, min_dist: float,
            min_cluster_size: int, min_samples: int) -> ClusterAssignment:
        embeddings_st = StandardScaler().fit_transform(embeddings)

        self.umap = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
        embeddings_umap = self.umap.fit_transform(embeddings_st)

        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True)
        self.hdbscan.fit(embeddings_umap)

        return ClusterAssignment(labels=self.hdbscan.labels_, probabilities=self.hdbscan.probabilities_)

    def assign(self, embeddings: np.ndarray):
        embeddings_umap = self.umap.transform(embeddings)
        labels, probabilities = hdbscan.approximate_predict(self.hdbscan, embeddings_umap)

        return ClusterAssignment(labels=labels, probabilities=probabilities)
