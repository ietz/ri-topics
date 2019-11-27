from typing import List, Dict

import numpy as np

from ri_topics.clustering import ClusterAssignment
from ri_topics.preprocessing import Document


class Summarizer:
    def __init__(self):
        self._representatives: Dict[int, Document] = {}

    def fit(self, docs: List[Document], assignment: ClusterAssignment):
        n_clusters = assignment.labels.max() + 1
        for label in range(n_clusters):
            cluster_mask = assignment.labels == label
            cluster_probs = assignment.probabilities[cluster_mask]
            cluster_representative_idx = np.where(cluster_mask)[cluster_probs.argmax()]
            self._representatives[label] = docs[cluster_representative_idx]

    def cluster_summary(self, label: int) -> Document:
        return self._representatives[label]
