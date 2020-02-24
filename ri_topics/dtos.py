from dataclasses import dataclass
from typing import List, Optional


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
    name: Optional[str]
    representative: Representative
    member_ids: List[int]

    @staticmethod
    def from_df_tuple(t, member_ids: List[int]):
        return Topic(
            topic_id=t.Index,
            name=t.name,
            representative=Representative.from_df_tuple(t),
            member_ids=member_ids,
        )
