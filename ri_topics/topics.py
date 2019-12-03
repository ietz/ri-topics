import dataclasses
import logging
from typing import List, Optional

import pandas as pd
import pandas.io.json

from ri_topics.clustering import Clusterer
from ri_topics.embedder import Embedder
from ri_topics.models import Tweet
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter

logger = logging.getLogger(__name__)


def select_representatives(tweet_df: pd.DataFrame) -> pd.DataFrame:
    labeled_tweets = tweet_df[tweet_df['label'] >= 0]
    representative_idxs = labeled_tweets.groupby('label')['probability'].idxmax()
    return tweet_df.iloc[representative_idxs].set_index('label')


def tweets_to_df(tweets: List[Tweet]):
    df = pd.io.json.json_normalize([dataclasses.asdict(tweet) for tweet in tweets])

    return pd.DataFrame({
        'status_id': df['status_id'],
        'created_at': pd.to_datetime(df['created_at_full']),
    })


class TopicModel:
    def __init__(self, account_name, embedder: Embedder, storage: RiStorageTwitter):
        self.account_name = account_name
        self.storage = storage
        self.embedder = embedder

        self.clusterer = Clusterer()
        self.tweet_df: Optional[pd.DataFrame] = None
        self.repr_df: Optional[pd.DataFrame] = None

    def train(self, **clusterer_kwargs):
        logger.info(f'Fetching tweets for {self.account_name}')
        tweets = self.storage.get_all_tweets_by_account_name(self.account_name)
        logger.info(f'Retrieved {len(tweets)} tweets')
        logger.info(f'Embedding tweets')
        embeddings = self.embedder.embed_tweets(tweets)
        logger.info(f'Building clustering model')
        assignment = self.clusterer.fit(embeddings, **clusterer_kwargs)

        logger.info(f'Processing clusters')
        self.tweet_df = tweets_to_df(tweets)
        self.tweet_df['label'] = assignment.labels
        self.tweet_df['probability'] = assignment.probabilities

        self.repr_df = select_representatives(self.tweet_df)
