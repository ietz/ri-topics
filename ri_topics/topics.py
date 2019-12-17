import dataclasses
import pickle
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
import pandas.io.json
from loguru import logger

from ri_topics.clustering import Clusterer, ClusterAssignment
from ri_topics.config import MODEL_DIR
from ri_topics.embedder import Embedder
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter, Tweet
from ri_topics.util import is_between, df_without, series_to_namedtuple


def select_representatives(tweet_df: pd.DataFrame) -> pd.DataFrame:
    labeled_tweets = tweet_df[tweet_df['label'] >= 0]
    representative_idxs = labeled_tweets.groupby('label')['probability'].idxmax()
    representatives = tweet_df.loc[representative_idxs]
    return representatives \
        .reset_index() \
        .rename(columns={'status_id': 'representative_id'}) \
        .set_index('label')


def tweets_to_df(tweets: List[Tweet]):
    df = pd.io.json.json_normalize([dataclasses.asdict(tweet) for tweet in tweets])
    df['sentiment'] = df['sentiment'].astype('category')
    df['tweet_class'] = df['tweet_class'].astype('category')
    df['created_at'] = pd.to_datetime(df['created_at_full'])
    df = df.drop(columns=['created_at_full'])
    with_idx = df.set_index('status_id')
    return with_idx


class TopicModel:
    persisted_tweet_attributes = ['label',  'probability'] + ['created_at']
    persisted_representative_attributes = ['representative_id'] + ['text']

    def __init__(self, account_name, clusterer_factory: Callable[[], Clusterer] = Clusterer):
        self.account_name = account_name

        self.clusterer = clusterer_factory()
        self.tweet_df: Optional[pd.DataFrame] = None
        self.repr_df: Optional[pd.DataFrame] = None

    def train(self, embedder: Embedder, storage: RiStorageTwitter):
        logger.info(f'Training model {self.account_name}')

        full_tweet_df = self._get_new_tweets(storage)
        labeled_tweet_df = self._process_tweets(full_tweet_df, embedder, assign=self.clusterer.fit)
        self.tweet_df = labeled_tweet_df[TopicModel.persisted_tweet_attributes]
        self.repr_df = select_representatives(labeled_tweet_df)[TopicModel.persisted_representative_attributes]

        n_assigned = np.sum(self.tweet_df['label'] >= 0)
        logger.info(f'Assigned {n_assigned} ({n_assigned/len(self.tweet_df):0.01%}) tweets '
                    f'into {len(self.repr_df)} clusters')

    def update(self, embedder: Embedder, storage: RiStorageTwitter):
        logger.info(f'Predicting new tweets for {self.account_name}')

        full_tweet_df = self._get_new_tweets(storage)
        update_df = self._process_tweets(full_tweet_df, embedder, assign=self.clusterer.predict)
        self._log_assignment_rate(update_df)
        self.tweet_df = self.tweet_df.append(update_df[TopicModel.persisted_tweet_attributes])

    def _get_new_tweets(self, storage: RiStorageTwitter) -> pd.DataFrame:
        logger.info(f'Fetching tweets for {self.account_name}')
        tweets = storage.get_all_tweets_by_account_name(self.account_name)
        df = df_without(tweets_to_df(tweets), self.tweet_df)
        logger.info(f'Retrieved {len(df)} new tweets')

        df_en = df[df['lang'] == 'en']
        n_discarded = len(df) - len(df_en)
        logger.info(f'Discarding {n_discarded} ({n_discarded / len(df):0.01%}) non-english tweets')
        return df_en

    def _process_tweets(self, full_tweet_df: pd.DataFrame, embedder: Embedder, assign: Callable[[np.ndarray], ClusterAssignment]) -> pd.DataFrame:
        embeddings = embedder.embed_texts(full_tweet_df['text'])
        logger.info('Assigning tweets to clusters')
        assignment = assign(embeddings)

        logger.info(f'Processing clusters')
        update_df = full_tweet_df.copy()
        update_df['label'] = assignment.labels
        update_df['probability'] = assignment.probabilities

        return update_df

    def count_tweets_by_topic(self, start_ts=None, end_ts=None) -> pd.DataFrame:
        tweets = self._tweets_in_time_range(start_ts, end_ts)
        tweet_counts = tweets.groupby('label').size().rename('tweet_count')
        act = self.repr_df.join(tweet_counts, how='outer')
        act['tweet_count'] = act['tweet_count'].fillna(0)
        return act

    def topic_by_id(self, topic_id) -> namedtuple:
        return series_to_namedtuple(topic_id, self.repr_df.loc[topic_id])

    def _tweets_in_time_range(self, start_ts, end_ts) -> pd.DataFrame:
        mask = is_between(self.tweet_df['created_at'], start_ts, end_ts)
        return self.tweet_df[mask]

    def _log_assignment_rate(self, df: pd.DataFrame):
        n_unassigned = np.sum(df['label'] == -1)
        pct_unassigned = n_unassigned / len(df)
        pct_unassigned_before = np.mean(self.tweet_df['label'] == -1)
        logger.info(
            f'{n_unassigned} ({pct_unassigned:0.01%}) new tweets are not assigned to a cluster '
            f'compared to {pct_unassigned_before:0.01%} of previous tweets'
        )


class TopicModelManager:
    def __init__(self, embedder: Embedder, storage: RiStorageTwitter):
        self.models = {}
        self.embedder = embedder
        self.storage = storage

    def get(self, account_name: str) -> TopicModel:
        if account_name not in self.models:
            if self._is_persisted(account_name):
                self._cache(self._load(account_name))
            else:
                self.save(self._build(account_name))

        return self.models[account_name]

    def save(self, model: TopicModel):
        self._cache(model)
        self._persist(model)

    def prepare_all(self):
        for name in self.model_names:
            self.get(name)  # "touch" the model to initialize it

    def update_all(self):
        for name in self.model_names:
            self.save(self._update(name))

    @property
    def model_names(self) -> List[str]:
        return self.storage.get_all_account_names()

    def _build(self, account_name: str) -> TopicModel:
        logger.info(f'Building model for {account_name}')
        model = TopicModel(account_name)
        model.train(embedder=self.embedder, storage=self.storage)
        return model

    def _update(self, account_name: str) -> TopicModel:
        model = self.get(account_name)
        model.update(self.embedder, self.storage)
        return model

    def _cache(self, model: TopicModel):
        self.models[model.account_name] = model

    def _persist(self, model: TopicModel):
        logger.info(f'Persisting model for {model.account_name}')
        with self._path(model.account_name).open(mode='wb+') as f:
            pickle.dump(model, f)

    def _load(self, account_name: str) -> TopicModel:
        logger.info(f'Loading persisted model for {account_name}')
        with self._path(account_name).open(mode='rb') as f:
            return pickle.load(f)

    def _is_persisted(self, account_name: str) -> bool:
        return self._path(account_name).exists()

    def _path(self, account_name: str) -> Path:
        return MODEL_DIR / f'{account_name}.pickle'
