import dataclasses
import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pandas.io.json
from loguru import logger

from ri_topics.clustering import Clusterer
from ri_topics.config import MODEL_DIR
from ri_topics.embedder import Embedder
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter, Tweet
from ri_topics.util import is_between


def select_representatives(tweet_df: pd.DataFrame) -> pd.DataFrame:
    labeled_tweets = tweet_df[tweet_df['label'] >= 0]
    representative_idxs = labeled_tweets.groupby('label')['probability'].idxmax()
    representatives = tweet_df.loc[representative_idxs]
    return pd.DataFrame({
        'label': representatives['label'],
        'representative_id': representatives.index.to_series(),
    }).set_index('label')


def tweets_to_df(tweets: List[Tweet]):
    df = pd.io.json.json_normalize([dataclasses.asdict(tweet) for tweet in tweets])

    return pd.DataFrame({
        'status_id': df['status_id'],
        'created_at': pd.to_datetime(df['created_at_full']),
    }).set_index('status_id')


class TopicModel:
    def __init__(self, account_name):
        self.account_name = account_name

        self.clusterer = Clusterer()
        self.tweet_df: Optional[pd.DataFrame] = None
        self.created_index: Optional[pd.DatetimeIndex] = None
        self.repr_df: Optional[pd.DataFrame] = None

    def train(self, embedder: Embedder, storage: RiStorageTwitter, **clusterer_kwargs):
        logger.info(f'Fetching tweets for {self.account_name}')
        tweets = storage.get_all_tweets_by_account_name(self.account_name)
        logger.info(f'Retrieved {len(tweets)} tweets')
        embeddings = embedder.embed_tweets(tweets)
        logger.info(f'Building clustering model')
        assignment = self.clusterer.fit(embeddings, **clusterer_kwargs)

        logger.info(f'Processing clusters')
        self.tweet_df = tweets_to_df(tweets)
        self.tweet_df['label'] = assignment.labels
        self.tweet_df['probability'] = assignment.probabilities

        self.repr_df = select_representatives(self.tweet_df)

    def count_tweets_by_topic(self, start_ts=None, end_ts=None) -> pd.DataFrame:
        mask = is_between(self.tweet_df['created_at'], start_ts, end_ts)
        tweets = self.tweet_df[mask]
        tweet_counts = tweets.groupby('label').size().rename('tweet_count')
        act = self.repr_df.join(tweet_counts, how='outer')
        act['tweet_count'] = act['tweet_count'].fillna(0)
        return act


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

    def _build(self, account_name: str) -> TopicModel:
        logger.info(f'Building model for {account_name}')
        model = TopicModel(account_name)
        model.train(embedder=self.embedder, storage=self.storage, n_components=10, n_neighbors=40, min_dist=0, min_cluster_size=30, min_samples=20)
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
