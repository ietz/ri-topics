from dataclasses import dataclass
from typing import List


@dataclass
class Representative:
    status_id: str
    text: str

    @staticmethod
    def from_df_tuple(t):
        return Representative(
            status_id=t.representative_id,
            text=t.text,
        )


@dataclass
class Topic:
    topic_id: str
    representative: Representative


@dataclass
class TopicContent(Topic):
    member_ids: List[int]

    @staticmethod
    def from_df_tuple(t, member_ids: List[int]):
        return TopicContent(
            topic_id=t.Index,
            representative=Representative.from_df_tuple(t),
            member_ids=member_ids,
        )


@dataclass
class TopicActivity(Topic):
    occurrences: int

    @staticmethod
    def from_df_tuple(t):
        return TopicActivity(
            topic_id=t.Index,
            representative=Representative.from_df_tuple(t),
            occurrences=t.tweet_count,
        )


@dataclass
class TrendOccurrences:
    before: int
    current: int


@dataclass
class Trend(Topic):
    score: float
    occurrences: TrendOccurrences

    @staticmethod
    def from_df_tuple(t):
        return Trend(
            topic_id=t.Index,
            representative=Representative.from_df_tuple(t),
            score=t.score,
            occurrences=TrendOccurrences(
                before=t.before_count,
                current=t.current_count,
            )
        )
