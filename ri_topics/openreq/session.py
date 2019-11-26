from requests import Session

from ri_topics.util import subpath_join


class OpenReqServiceSession(Session):
    def __init__(self, base_url, bearer_token):
        super().__init__()
        self.base_url = base_url
        self.headers.update({
            'Authorization': f'Bearer {bearer_token}',
        })

    def request(self, method, url, verify=False, *args, **kwargs):
        full_url = subpath_join(self.base_url, url)
        return super().request(method, full_url, verify=verify, *args, **kwargs)
