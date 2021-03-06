import dataclasses
import pickle
import threading
import time
from pathlib import Path
from schedule import Scheduler
from typing import List, Optional, Callable, Dict

import numpy as np
import pandas as pd
import pandas.io.json
from loguru import logger

from ri_topics.clustering import Clusterer, ClusterAssignment
from ri_topics.config import MODEL_DIR
from ri_topics.embedder import Embedder
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter, Tweet
from ri_topics.util import df_without, default_value, pct


def select_representatives(tweet_df: pd.DataFrame) -> pd.DataFrame:
    labeled_tweets = tweet_df[tweet_df['label'] >= 0]
    representative_idxs = labeled_tweets.groupby('label')['probability'].idxmax()
    representatives = tweet_df.loc[representative_idxs]
    return representatives \
        .rename_axis('representative_id') \
        .reset_index() \
        .set_index('label')


def tweets_to_df(tweets: List[Tweet]):
    def from_dicts(dicts: List[Dict]) -> pd.DataFrame:
        df = pd.io.json.json_normalize(dicts)
        df['sentiment'] = df['sentiment'].astype('category')
        df['tweet_class'] = df['tweet_class'].astype('category')
        df['created_at'] = pd.to_datetime(df['created_at_full'])
        df = df.drop(columns=['created_at_full'])
        with_idx = df.set_index('status_id')
        return with_idx

    if len(tweets) > 0:
        return from_dicts([dataclasses.asdict(tweet) for tweet in tweets])
    else:
        # generate empty df by using the default values of Tweet dataclass for field and types
        return from_dicts([default_value(Tweet)]).iloc[:0]


class TopicModel:
    persisted_tweet_attributes = ['label',  'probability'] + []
    persisted_representative_attributes = ['representative_id'] + ['text', 'name']

    def __init__(self, account_name, clusterer_factory: Callable[[], Clusterer] = Clusterer):
        self.account_name = account_name

        self.clusterer = clusterer_factory()
        self.tweet_df: Optional[pd.DataFrame] = None
        self.topic_df: Optional[pd.DataFrame] = None

    def train(self, embedder: Embedder, storage: RiStorageTwitter):
        logger.info(f'Training model {self.account_name}')

        full_tweet_df = self._get_new_tweets(storage)
        labeled_tweet_df = self._process_tweets(full_tweet_df, embedder, assign=self.clusterer.fit)
        self.tweet_df = labeled_tweet_df[TopicModel.persisted_tweet_attributes]

        topic_df = select_representatives(labeled_tweet_df)
        topic_df['name'] = None
        self.topic_df = topic_df[TopicModel.persisted_representative_attributes]

        n_assigned = np.sum(self.tweet_df['label'] >= 0)
        logger.info(f'Assigned {n_assigned} ({n_assigned/len(self.tweet_df):0.01%}) tweets '
                    f'into {len(self.topic_df)} clusters')

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

        return df

    def _process_tweets(self, full_tweet_df: pd.DataFrame, embedder: Embedder, assign: Callable[[np.ndarray], ClusterAssignment]) -> pd.DataFrame:
        language_mask = full_tweet_df['lang'] == 'en'
        account_mask = full_tweet_df['user_name'] != self.account_name
        filtered_tweet_df = full_tweet_df[language_mask & account_mask]
        n_discarded = len(full_tweet_df) - len(filtered_tweet_df)
        logger.info(f'Discarding {n_discarded} ({pct(n_discarded, len(full_tweet_df)):0.01%}) tweets')

        embeddings = embedder.embed_texts(filtered_tweet_df['text'])
        logger.info('Assigning tweets to clusters')
        assignment = assign(embeddings)

        logger.info(f'Processing clusters')
        update_df = filtered_tweet_df.copy()
        update_df['label'] = assignment.labels
        update_df['probability'] = assignment.probabilities

        return update_df

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

    def schedule_updates(self) -> threading.Event:
        scheduler = Scheduler()
        scheduler.every().day.at('04:30').do(self.update_all)

        cease_run = threading.Event()

        class ScheduleThread(threading.Thread):
            def run(self) -> None:
                while not cease_run.is_set():
                    scheduler.run_pending()
                    time.sleep(1)

        schedule_thread = ScheduleThread()
        schedule_thread.start()

        return cease_run

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
