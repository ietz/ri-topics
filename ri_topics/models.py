from dataclasses import dataclass


@dataclass
class Topic:
    topic_id: str
    representative_id: str


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
            representative_id=t.representative_id,
            score=t.score,
            occurrences=TrendOccurrences(
                before=t.before_count,
                current=t.current_count,
            )
        )


@dataclass
class TopicActivity(Topic):
    occurrences: int

    @staticmethod
    def from_df_tuple(t):
        return TopicActivity(
            topic_id=t.Index,
            representative_id=t.representative_id,
            occurrences=t.tweet_count,
        )
