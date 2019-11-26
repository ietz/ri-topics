from dataclasses import dataclass
from typing import Optional


@dataclass
class Tweet:
    username: str
    text: str


@dataclass
class Topic:
    id: str
    name: Optional[str]
    representative: Tweet


@dataclass
class OccurrenceCount:
    current: int
    previous: int


@dataclass
class Trend(Topic):
    trendiness_score: float
    occurrences: OccurrenceCount
