import dataclasses
import pickle
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
from ri_topics.util import is_between, df_without


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
    df['sentiment'] = df['sentiment'].astype('category')
    df['tweet_class'] = df['tweet_class'].astype('category')
    df['created_at'] = pd.to_datetime(df['created_at_full'])
    df = df.drop(columns=['created_at_full'])
    with_idx = df.set_index('status_id')
    return with_idx


class TopicModel:
    persisted_tweet_attributes = ['created_at']

    def __init__(self, account_name):
        self.account_name = account_name

        self.clusterer = Clusterer()
        self.tweet_df: Optional[pd.DataFrame] = None
        self.created_index: Optional[pd.DatetimeIndex] = None
        self.repr_df: Optional[pd.DataFrame] = None

    def train(self, embedder: Embedder, storage: RiStorageTwitter, **clusterer_kwargs):
        logger.info(f'Training model {self.account_name}')

        def assign(embeddings: np.ndarray) -> ClusterAssignment:
            return self.clusterer.fit(embeddings, **clusterer_kwargs)

        self.tweet_df = self._process_new_tweets(embedder, storage, assign)
        self.repr_df = select_representatives(self.tweet_df)

    def update(self, embedder: Embedder, storage: RiStorageTwitter):
        logger.info(f'Predicting new tweets for {self.account_name}')

        update_df = self._process_new_tweets(embedder, storage, self.clusterer.predict)
        self._log_assignment_rate(update_df)
        self.tweet_df = self.tweet_df.append(update_df)

    def _process_new_tweets(self, embedder: Embedder, storage: RiStorageTwitter, assign: Callable[[np.ndarray], ClusterAssignment]) -> pd.DataFrame:
        logger.info(f'Fetching tweets for {self.account_name}')
        tweets = storage.get_all_tweets_by_account_name(self.account_name)
        full_tweets_df = df_without(tweets_to_df(tweets), self.tweet_df)
        logger.info(f'Retrieved {len(full_tweets_df)} new tweets')

        embeddings = embedder.embed_texts(full_tweets_df['text'])
        logger.info('Assigning tweets to clusters')
        assignment = assign(embeddings)

        logger.info(f'Processing clusters')
        update_df = full_tweets_df[TopicModel.persisted_tweet_attributes].copy()
        update_df['label'] = assignment.labels
        update_df['probability'] = assignment.probabilities

        return update_df

    def count_tweets_by_topic(self, start_ts=None, end_ts=None) -> pd.DataFrame:
        mask = is_between(self.tweet_df['created_at'], start_ts, end_ts)
        tweets = self.tweet_df[mask]
        tweet_counts = tweets.groupby('label').size().rename('tweet_count')
        act = self.repr_df.join(tweet_counts, how='outer')
        act['tweet_count'] = act['tweet_count'].fillna(0)
        return act

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
