import dataclasses
from typing import List, Any, Dict, Optional
from urllib.parse import urljoin

import pandas as pd


def force_trailing_slash(url: str) -> str:
    return url.rstrip('/') + '/'


def subpath_join(base_url: str, subpath: str) -> str:
    return urljoin(
        force_trailing_slash(base_url),
        subpath.lstrip('/')
    )


def init_from_dicts(dataclass, data_dicts: List[Dict[str, Any]]):
    fields = dataclasses.fields(dataclass)
    field_names = set([field.name for field in fields])

    nested_values = {field.name: init_from_dicts(field.type, [data_dict[field.name] for data_dict in data_dicts]) for field in fields if dataclasses.is_dataclass(field.type)}

    return [
        dataclass(**{key: (value if key not in nested_values else nested_values[key][idx]) for key, value in data_dict.items() if key in field_names})
        for idx, data_dict in enumerate(data_dicts)
    ]


def is_between(a, start=None, end=None):
    if start is None and end is None:
        # noinspection PyComparisonWithNone
        return True | (a == None)
    else:
        return (start is None or start <= a) & (end is None or a < end)


def df_without(left: pd.DataFrame, right: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Performs left outer exclusive join. Result contains all rows from the left df except for
    those which are also present in the right df."""
    if right is None:
        return left

    idxs = left.index.difference(right.index)
    return left.loc[idxs]
