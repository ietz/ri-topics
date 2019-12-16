from dataclasses import dataclass


@dataclass
class Representative:
    status_id: str
    text: str


@dataclass
class Topic:
    topic_id: str
    representative: Representative


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
            representative=Representative(
                status_id=t.representative_id,
                text=t.text,
            ),
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
            representative=Representative(
                status_id=t.representative_id,
                text=t.text,
            ),
            occurrences=t.tweet_count,
        )
