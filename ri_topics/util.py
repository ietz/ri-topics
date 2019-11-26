from urllib.parse import urljoin


def force_trailing_slash(url: str) -> str:
    return url.rstrip('/') + '/'


def subpath_join(base_url: str, subpath: str) -> str:
    return urljoin(
        force_trailing_slash(base_url),
        subpath.lstrip('/')
    )
