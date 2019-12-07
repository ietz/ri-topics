from dataclasses import dataclass
from typing import Optional

import numpy as np
import hdbscan
from loguru import logger
from sklearn.preprocessing import StandardScaler
import umap

from ri_topics.util import clamp


@dataclass
class ClusterAssignment:
    labels: np.ndarray
    probabilities: np.ndarray


@dataclass
class ClustererParams:
    n_components: int
    n_neighbors: int
    min_dist: float
    min_cluster_size: int
    min_samples: int

    @staticmethod
    def for_sample_size(n: int):
        return ClustererParams(
            n_components=clamp(1, 20, n-2),
            n_neighbors=clamp(1, 40, int(n*2/3)),
            min_dist=0.0,
            min_cluster_size=clamp(2, 30, int(n/20)),
            min_samples=clamp(1, 20, int(n/35)),
        )


class Clusterer:
    """Clustering using UMAP and HDBSCAN"""
    def __init__(self):
        self.umap: Optional[umap.UMAP] = None
        self.hdbscan: Optional[hdbscan.HDBSCAN] = None

    @property
    def is_fitted(self) -> bool:
        return self.umap is not None and self.hdbscan is not None

    def fit(self, embeddings: np.ndarray) -> ClusterAssignment:
        if len(embeddings) <= 1:
            logger.warning(f'Not fitting clusterer because too few embeddings are provided ({len(embeddings)})')
            return Clusterer._empty_assignment(len(embeddings))

        params = ClustererParams.for_sample_size(len(embeddings))
        embeddings_st = StandardScaler().fit_transform(embeddings)

        logger.info('Fitting UMAP')
        self.umap = umap.UMAP(n_components=params.n_components, n_neighbors=params.n_neighbors, min_dist=params.min_dist)
        embeddings_umap = self.umap.fit_transform(embeddings_st)

        logger.info('Running HDBSCAN')
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=params.min_cluster_size, min_samples=params.min_samples, prediction_data=True)
        self.hdbscan.fit(embeddings_umap)

        return ClusterAssignment(labels=self.hdbscan.labels_, probabilities=self.hdbscan.probabilities_)

    def predict(self, embeddings: np.ndarray):
        if not self.is_fitted:
            return Clusterer._empty_assignment(len(embeddings))

        embeddings_umap = self.umap.transform(embeddings)
        labels, probabilities = hdbscan.approximate_predict(self.hdbscan, embeddings_umap)

        return ClusterAssignment(labels=labels, probabilities=probabilities)

    @staticmethod
    def _empty_assignment(n: int = 0):
        return ClusterAssignment(
            labels=np.full(n, fill_value=-1, dtype=np.int),
            probabilities=np.zeros(n, dtype=np.float),
        )
