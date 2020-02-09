import numpy as np
import pandas as pd

from ri_topics.topics import TopicModel


def find_top(model: TopicModel, start, end) -> pd.DataFrame:
    return model.count_tweets_by_topic(start, end).loc[0:]


def find_trends(model: TopicModel, start, end) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz='UTC')
    end_ts = pd.Timestamp(end, tz='UTC')
    window_length = end_ts - start_ts
    before_ts = start_ts - window_length

    before = model.count_tweets_by_topic(before_ts, start_ts)['tweet_count']
    current = model.count_tweets_by_topic(start_ts, end_ts)['tweet_count']
    score = current - before

    return model.topic_df.join([
        before.rename('before_count'),
        current.rename('current_count'),
        score.rename('score'),
    ])
