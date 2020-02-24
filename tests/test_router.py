import unittest
from unittest.mock import Mock
from unittest import mock

import pandas as pd

from ri_topics.router import app
from ri_topics.topics import TopicModelManager, TopicModel, pickle as topic_pickle


def get_dummy_topic_model(*args, **kwargs):
    model = TopicModel(account_name='FitbitSupport')
    model.tweet_df = pd.DataFrame(
        columns=['label', 'probability'],
        data=[
            [ 0, 1.0],
            [ 0, 0.7],
            [ 1, 1.0],
            [ 1, 0.4],
            [ 2, 1.0],
            [-1, 0.0],
        ],
        index=['0', '1', '10', '11', '20', '90'],
    ).rename_axis('status_id')
    model.topic_df = pd.DataFrame(
        columns=['representative_id', 'text', 'name'],
        data=[
            [ '0', 'Text for cluster 0', None],
            ['10', 'Text for cluster 1', 'Name for cluster 1'],
            ['20', 'Text for cluster 2', None],
        ],
        index=[0, 1, 2],
    ).rename_axis('label')
    return model


@mock.patch.object(topic_pickle, 'load', get_dummy_topic_model)
class TestRestEndpoint(unittest.TestCase):
    def setUp(self):
        embedder = Mock()
        storage = Mock(account_names=['FitbitSupport'])
        manager = TopicModelManager(embedder=embedder, storage=storage)
        manager_path_patch = mock.patch.object(manager, '_path')
        manager_path_patch.start()
        self.addCleanup(manager_path_patch.stop)

        app.model_manager = manager
        self.client = app.test_client()

    def test_topics(self):
        resp = self.client.get('/FitbitSupport/topics/')
        self.assertEqual(
            resp.json,
            [
                {'topic_id': 0, 'representative': {'status_id':  '0', 'text': 'Text for cluster 0'}, 'member_ids': ['0', '1'], 'name': None},
                {'topic_id': 1, 'representative': {'status_id':  '10', 'text': 'Text for cluster 1'}, 'member_ids': ['10', '11'], 'name': 'Name for cluster 1'},
                {'topic_id': 2, 'representative': {'status_id':  '20', 'text': 'Text for cluster 2'}, 'member_ids': ['20'], 'name': None},
            ]
        )


if __name__ == '__main__':
    unittest.main()
