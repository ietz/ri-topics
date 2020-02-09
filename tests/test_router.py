import unittest
from unittest.mock import Mock
from unittest import mock

import pandas as pd

from ri_topics.router import app
from ri_topics.topics import TopicModelManager, TopicModel, pickle as topic_pickle


def get_dummy_topic_model(*args, **kwargs):
    model = TopicModel(account_name='FitbitSupport')
    model.tweet_df = pd.DataFrame(
        columns=['label', 'probability', 'created_at'],
        data=[
            [ 0, 1.0, pd.to_datetime('2019-12-31 23:59:59 UTC')],
            [ 0, 0.7, pd.to_datetime('2019-12-30 13:56:21 UTC')],
            [ 1, 1.0, pd.to_datetime('2019-12-30 17:14:54 UTC')],
            [ 1, 0.4, pd.to_datetime('2019-12-29 07:31:22 UTC')],
            [ 2, 1.0, pd.to_datetime('2019-12-28 14:33:16 UTC')],
            [-1, 0.0, pd.to_datetime('2019-12-30 21:48:38 UTC')],
        ],
        index=['0', '1', '10', '11', '20', '90'],
    )
    model.topic_df = pd.DataFrame(
        columns=['representative_id', 'text', 'name'],
        data=[
            [ '0', 'Text for cluster 0', None],
            ['10', 'Text for cluster 1', 'Name for cluster 1'],
            ['20', 'Text for cluster 2', None],
        ],
        index=[0, 1, 2],
    )
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

        self.topic_data = {
            0: {'topic_id': 0, 'representative': {'status_id':  '0', 'text': 'Text for cluster 0'}, 'name': None},
            1: {'topic_id': 1, 'representative': {'status_id':  '10', 'text': 'Text for cluster 1'}, 'name': 'Name for cluster 1'},
            2: {'topic_id': 2, 'representative': {'status_id':  '20', 'text': 'Text for cluster 2'}, 'name': None},
        }

        app.model_manager = manager
        self.client = app.test_client()

    def test_frequent(self):
        resp = self.client.get('/FitbitSupport/frequent')
        self.assertEqual(
            resp.json,
            [{**self.topic_data[0], 'occurrences': 2},
             {**self.topic_data[1], 'occurrences': 2},
             {**self.topic_data[2], 'occurrences': 1}]
        )

    def test_frequent_range(self):
        resp = self.client.get('/FitbitSupport/frequent?start=2019-12-30&end=2020-01-01')
        self.assertEqual(
            resp.json,
            [{**self.topic_data[0], 'occurrences': 2},
             {**self.topic_data[1], 'occurrences': 1},
             {**self.topic_data[2], 'occurrences': 0}]
        )

    def test_trends(self):
        resp = self.client.get('/FitbitSupport/trends?start=2019-12-30&end=2020-01-01')
        all_expected_ordered = [
            {**self.topic_data[0], 'occurrences': {'before': 0, 'current': 2}},  # increasing
            {**self.topic_data[1], 'occurrences': {'before': 1, 'current': 1}},  # constant
            {**self.topic_data[2], 'occurrences': {'before': 1, 'current': 0}},  # decreasing
        ]

        for all_expected, all_actual in [(all_expected_ordered, resp.json['rising']), (all_expected_ordered[::-1], resp.json['falling'])]:
            for expected, actual in zip(all_expected, all_actual):  # explicitly does not
                self.assertIn('score', actual.keys())  # doesn't have an expected value

                actual_subdict = {key: actual[key] for key in expected.keys()}
                self.assertEqual(expected, actual_subdict)

    def test_topic(self):
        resp = self.client.get('/FitbitSupport/topics/0')
        self.assertEqual(
            resp.json,
            {**self.topic_data[0], 'member_ids': ['0', '1']},
        )


if __name__ == '__main__':
    unittest.main()
