import unittest
from typing import List
from unittest import mock
from unittest.mock import Mock

import numpy as np

from ri_topics.clustering import Clusterer, ClusterAssignment
from ri_topics.embedder import Embedder
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter, Tweet
from ri_topics.topics import TopicModel
from ri_topics.util import mock_dataclass_asdict

embedding_dim = 768
account_names = ['FitbitSupport']
all_tweets = [
    Mock(spec=Tweet, **{'status_id': str(idx), 'created_at_full': date, 'lang': 'en', 'text': f'{idx}'})
    for idx, date in enumerate([
        'Tue Dec 17 07:25:24 +0000 2019',
        'Tue Dec 17 03:08:09 +0000 2019',
        'Mon Dec 16 20:05:09 +0000 2019',
        'Mon Dec 16 19:45:14 +0000 2019',
        'Mon Dec 16 15:45:25 +0000 2019',
        'Mon Dec 16 03:01:32 +0000 2019',
    ])
]
initial_tweets = [all_tweets[i] for i in [0, 1, 3]]
update_tweets = [all_tweets[i] for i in [1, 2, 3, 4, 5]]
labels = np.array([-1,  0,  0,  1,  1, -1, -1])
probs =  np.array([0., 1., 1., 1., 1., 0., 0.])


def mock_embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts as their status_id, which is encoded as the text.
    E.g. text 3 is embedded as [3]*embedding_dim"""

    return np.array([int(t) for t in texts]) \
        .repeat(embedding_dim) \
        .reshape((-1, embedding_dim))


def mock_cluster(embeddings: np.ndarray) -> ClusterAssignment:
    status_ids = embeddings[:, 0]
    return ClusterAssignment(
        labels=labels[status_ids],
        probabilities=probs[status_ids],
    )


class TestTopicModel(unittest.TestCase):
    @mock.patch('ri_topics.topics.dataclasses')
    def test_train_and_update(self, dataclasses):
        dataclasses.asdict.side_effect = mock_dataclass_asdict

        storage = Mock(spec=RiStorageTwitter, **{
            'get_all_account_names.return_value': account_names,
            'get_all_tweets_by_account_name.side_effect': [initial_tweets, update_tweets],
        })
        embedder = Mock(spec=Embedder, **{
            'embed_texts.side_effect': mock_embed_texts,
        })
        clusterer = Mock(spec=Clusterer, **{
            'fit.side_effect': mock_cluster,
            'predict.side_effect': mock_cluster,
        })
        clusterer_factory = Mock(return_value=clusterer)

        topic_model = TopicModel('FitbitSupport', clusterer_factory=clusterer_factory)

        # Initial training
        topic_model.train(embedder, storage)
        self.assertSetEqual({'0', '1', '3'}, set(topic_model.tweet_df.index))
        self.assertSetEqual({0, 1}, set(topic_model.repr_df.index))

        # Update
        topic_model.update(embedder, storage)
        self.assertSetEqual({'0', '1', '2', '3', '4', '5'}, set(topic_model.tweet_df.index))


if __name__ == '__main__':
    unittest.main()
