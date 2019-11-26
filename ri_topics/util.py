import dataclasses
from typing import List, Any, Dict
from urllib.parse import urljoin


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
        dataclass(**{key: (value if key not in nested_values else nested_values[key][idx]) for idx, (key, value) in enumerate(data_dict.items()) if key in field_names})
        for data_dict in data_dicts
    ]
