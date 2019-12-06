import numpy as np
import pandas as pd

from ri_topics.topics import TopicModel


def find_trends(model: TopicModel, start, end):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    window_length = end_ts - start_ts
    before_ts = start_ts - window_length

    before = model.count_tweets_by_topic(before_ts, start_ts)['tweet_count']
    current = model.count_tweets_by_topic(start_ts, end_ts)['tweet_count']
    size = pd.concat([current, before], axis=1).max(axis=1)
    increase = current - before  # possibly try clamping (e.g. sigmoid-esque)
    importance = np.log10(1 + size)  # possibly try other bases instead of 10
    score = (increase / size) * importance

    return model.repr_df.join([
        before.rename('before_count'),
        current.rename('current_count'),
        score.rename('score'),
    ])
