import unittest
from unittest import mock
from unittest.mock import Mock

import numpy as np

from ri_topics.clustering import Clusterer, ClustererParams

embedding_dim = 768
n_labels = 5


class TestClusterer(unittest.TestCase):
    @mock.patch('ri_topics.clustering.hdbscan')
    @mock.patch('ri_topics.clustering.umap')
    def test_fit(self, umap, hdbscan):
        n_samples = 100

        labels = np.arange(n_samples) % n_labels
        probabilities = np.random.random(n_samples)
        embeddings = np.random.random((n_samples, embedding_dim))

        umap.UMAP.side_effect = lambda n_components, *args, **kwargs: Mock(**{
            'fit_transform.return_value': np.random.random((n_samples, n_components))
        })
        hdbscan.HDBSCAN.return_value.configure_mock(**{
            'labels_': labels,
            'probabilities_': probabilities,
        })

        clusterer = Clusterer()
        fit_assignment = clusterer.fit(embeddings)
        np.testing.assert_equal(fit_assignment.labels, labels)
        np.testing.assert_equal(fit_assignment.probabilities, probabilities)

    @mock.patch('ri_topics.clustering.hdbscan')
    @mock.patch('ri_topics.clustering.umap')
    def test_predict(self, umap, hdbscan):
        n_samples = 100

        labels = np.arange(n_samples) % n_labels
        probabilities = np.random.random(n_samples)
        embeddings = np.random.random((n_samples, embedding_dim))

        umap.UMAP.side_effect = lambda n_components, *args, **kwargs: Mock(**{
            'transform.return_value': np.random.random((n_samples, n_components))
        })
        hdbscan.approximate_predict.return_value = labels, probabilities

        clusterer = Clusterer()
        clusterer.hdbscan = hdbscan.HDBSCAN()
        clusterer.umap = umap.UMAP(n_components=ClustererParams.for_sample_size(n_samples).n_components)
        fit_assignment = clusterer.predict(embeddings)
        np.testing.assert_equal(fit_assignment.labels, labels)
        np.testing.assert_equal(fit_assignment.probabilities, probabilities)

    @mock.patch('ri_topics.clustering.hdbscan')
    @mock.patch('ri_topics.clustering.umap')
    def test_fit_empty_data(self, umap, hdbscan):
        clusterer = Clusterer()

        fit_assignment = clusterer.fit(np.empty((1, embedding_dim), dtype=np.float))
        np.testing.assert_equal(fit_assignment.labels, [-1])
        np.testing.assert_equal(fit_assignment.probabilities, [0.])
        self.assertFalse(clusterer.is_fitted)

    @mock.patch('ri_topics.clustering.hdbscan')
    @mock.patch('ri_topics.clustering.umap')
    def test_predict_not_fitted(self, umap, hdbscan):
        clusterer = Clusterer()

        fit_assignment = clusterer.predict(np.random.random((10, embedding_dim)))
        np.testing.assert_equal(fit_assignment.labels, [-1] * 10)
        np.testing.assert_equal(fit_assignment.probabilities, [0.] * 10)


if __name__ == '__main__':
    unittest.main()
