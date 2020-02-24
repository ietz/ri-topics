import unittest
from typing import List
from unittest import mock
from unittest.mock import Mock

import numpy as np

from ri_topics.clustering import Clusterer, ClusterAssignment
from ri_topics.embedder import Embedder
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter, Tweet
from ri_topics.topics import TopicModel, TopicModelManager
from ri_topics.util import mock_dataclass_asdict

embedding_dim = 768
account_names = ['FitbitSupport']
all_tweets = [
    Mock(spec=Tweet, **{'status_id': str(idx), 'lang': 'en', 'text': f'{idx}'})
    for idx in range(6)
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
        self.assertSetEqual({0, 1}, set(topic_model.topic_df.index))

        # Update
        topic_model.update(embedder, storage)
        self.assertSetEqual({'0', '1', '2', '3', '4', '5'}, set(topic_model.tweet_df.index))


class TestTopicModelManager(unittest.TestCase):
    @mock.patch('ri_topics.topics.pickle')
    @mock.patch('ri_topics.topics.TopicModel')
    @mock.patch.object(TopicModelManager, '_path')
    def test_create_model(self, mock_manager_path, MockTopicModel, mock_pickle):
        account_name = 'A'

        embedder = Mock(spec=Embedder)
        storage = Mock(spec=RiStorageTwitter, **{
            'get_all_account_names.return_value': [account_name],
        })

        MockTopicModel.return_value = Mock(**{
            'account_name': account_name,
        })
        mock_manager_path.return_value = Mock(**{
            'exists.return_value': False,
            'open': mock.mock_open(),
        })

        manager = TopicModelManager(embedder, storage)
        manager.prepare_all()
        self.assertEqual(0, mock_pickle.load.call_count)
        self.assertEqual(1, mock_pickle.dump.call_count)

    @mock.patch('ri_topics.topics.pickle')
    @mock.patch('ri_topics.topics.TopicModel')
    @mock.patch.object(TopicModelManager, '_path')
    def test_load_model(self, mock_manager_path, MockTopicModel, mock_pickle):
        account_name = 'A'

        embedder = Mock(spec=Embedder)
        storage = Mock(spec=RiStorageTwitter, **{
            'get_all_account_names.return_value': [account_name],
        })

        MockTopicModel.return_value = Mock(**{
            'account_name': account_name,
        })
        mock_manager_path.return_value = Mock(**{
            'exists.return_value': True,
            'open': mock.mock_open(),
        })
        mock_pickle.configure_mock(**{
            'load.return_value': MockTopicModel(),
        })

        manager = TopicModelManager(embedder, storage)
        manager.prepare_all()
        self.assertEqual(1, mock_pickle.load.call_count)
        self.assertEqual(0, mock_pickle.dump.call_count)

    @mock.patch('ri_topics.topics.pickle')
    @mock.patch('ri_topics.topics.TopicModel')
    @mock.patch.object(TopicModelManager, '_path')
    def test_update(self, mock_manager_path, MockTopicModel, mock_pickle):
        topic_models = {name: Mock(account_name=name) for name in ['A', 'B', 'C']}

        MockTopicModel.side_effect = lambda account_name: topic_models[account_name]
        mock_manager_path.side_effect = lambda account_name: Mock(**{
            'exists.return_value': True,
            'open': mock.mock_open(read_data=account_name),
        })
        mock_pickle.configure_mock(**{
            'load.side_effect': lambda f: topic_models[f.read()],
        })

        embedder = Mock(spec=Embedder)
        storage = Mock(spec=RiStorageTwitter, **{
            'get_all_account_names.return_value': list(topic_models.keys()),
        })

        manager = TopicModelManager(embedder, storage)
        manager.update_all()
        self.assertTrue(all([model.update.called for model in topic_models.values()]))
        self.assertEqual(len(topic_models), mock_pickle.load.call_count)
        self.assertEqual(len(topic_models), mock_pickle.dump.call_count)


if __name__ == '__main__':
    unittest.main()
