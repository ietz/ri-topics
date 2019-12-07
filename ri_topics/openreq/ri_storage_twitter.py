from dataclasses import dataclass
from typing import List

from ri_topics.openreq.session import OpenReqServiceSession
from ri_topics.util import init_from_dicts


@dataclass
class Topic:
    label: str
    score: int


@dataclass
class Topics:
    first_class: Topic
    second_class: Topic


@dataclass
class Tweet:
    created_at: int
    created_at_full: str
    favorite_count: int
    retweet_count: int
    text: str
    status_id: str
    user_name: str
    in_reply_to_screen_name: str
    hashtags: List[str]
    lang: str
    sentiment: str
    sentiment_score: int
    tweet_class: str
    classifier_certainty: int
    is_annotated: bool
    topics: Topics


class RiStorageTwitter:
    def __init__(self, base_url: str, bearer_token: str):
        self.session = OpenReqServiceSession(base_url, bearer_token)

    def get_all_account_names(self) -> List[str]:
        response = self.session.get(f'/account_name/all')
        all_names = response.json()['twitter_account_names']
        return [name for name in all_names if name]  # required to filter out invalid account ""

    def get_all_tweets_by_account_name(self, account_name: str) -> List[Tweet]:
        response = self.session.get(f'/account_name/{account_name}/all')
        return init_from_dicts(Tweet, response.json())
