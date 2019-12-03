import dataclasses
import logging
import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas.io.json import json_normalize
from tqdm import tqdm

from ri_topics.clustering import Clusterer
from ri_topics.embedder import Embedder
from ri_topics.models import Tweet
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter
from ri_topics.preprocessing import Document


def tweets_to_df(tweets: List[Tweet]) -> pd.DataFrame:
    df = json_normalize([dataclasses.asdict(tweet) for tweet in tweets])

    return pd.DataFrame({
        'status_id': df['status_id'],
        'created_at': pd.to_datetime(df['created_at_full']),
    })


def select_representatives(tweet_df: pd.DataFrame) -> pd.DataFrame:
    labeled_tweets = tweet_df[tweet_df['label'] >= 0]
    representative_idxs = labeled_tweets.groupby('label')['probability'].idxmax()
    return tweet_df.iloc[representative_idxs].set_index('label')


if __name__ == '__main__':
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    rist = RiStorageTwitter(
        base_url=os.getenv('RI_STORAGE_TWITTER_BASE_URL'),
        bearer_token=os.getenv('BEARER_TOKEN'),
    )

    logger.info('Fetching tweets')
    tweets = rist.get_all_tweets_by_account_name('FitbitSupport')
    logger.info(f'Retrieved {len(tweets)} tweets')

    texts = [tweet.text for tweet in tweets]
    docs = [Document(text) for text in tqdm(texts, desc='Preprocessing', unit='Tweets')]

    logger.info('Generating embeddings')
    embedder = Embedder()
    embedder.embed(docs)
    embeddings = np.array([doc.embedding for doc in docs])

    logger.info('Running topic clustering')
    clusterer = Clusterer()
    assignment = clusterer.fit(embeddings, n_components=10, n_neighbors=40, min_dist=0, min_cluster_size=30, min_samples=20)

    tweet_df = tweets_to_df(tweets)
    tweet_df['label'] = assignment.labels
    tweet_df['probability'] = assignment.probabilities

    repr_df = select_representatives(tweet_df)

    # logger.info('Starting server')
    # app.run(host='0.0.0.0', port='8888')
