import unittest

import requests_mock

from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter, Tweet, Topics, Topic

base_url = 'mock://base.url.com/subpath'
bearer_token = 'bearertoken'
request_headers = {'Authorization': f'Bearer {bearer_token}'}

account_name_response = '{"twitter_account_names": ["A"]}'
tweets_response = """
[{
  "created_at": 20191217,
  "created_at_full": "Tue Dec 17 15:04:10 +0000 2019",
  "favorite_count": 0,
  "retweet_count": 0,
  "text": "@FitbitSupport Ever since the firmware update my brand new Versa 2 won't go more than 30 minutes without freezing and rebooting.",
  "status_id": "1206953238974599168",
  "user_name": "MoonlightBard",
  "in_reply_to_screen_name": "FitbitSupport",
  "hashtags": [],
  "lang": "en",
  "sentiment": "NEUTRAL",
  "sentiment_score": 0,
  "tweet_class": "irrelevant",
  "classifier_certainty": -1,
  "is_annotated": false,
  "topics": {
    "first_class": {
      "label": "",
      "score": 0
    },
    "second_class": {
      "label": "",
      "score": 0
    }
  }
}]"""

tweets = [Tweet(
    created_at=20191217,
    created_at_full="Tue Dec 17 15:04:10 +0000 2019",
    favorite_count=0,
    retweet_count=0,
    text="@FitbitSupport Ever since the firmware update my brand new Versa 2 won't go more than 30 minutes without freezing and rebooting.",
    status_id="1206953238974599168",
    user_name="MoonlightBard",
    in_reply_to_screen_name="FitbitSupport",
    hashtags=[],
    lang="en",
    sentiment="NEUTRAL",
    sentiment_score=0,
    tweet_class="irrelevant",
    classifier_certainty=-1,
    is_annotated=False,
    topics=Topics(
        first_class=Topic(
            label="",
            score=0,
        ),
        second_class=Topic(
            label="",
            score=0,
        ),
    ),
)]


class RiStorageTwitterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = RiStorageTwitter(base_url=base_url, bearer_token=bearer_token)

    @requests_mock.mock()
    def test_account_names(self, req):
        req.get(base_url + '/account_name/all', text=account_name_response, request_headers=request_headers)
        self.assertEqual(['A'], self.storage.get_all_account_names())

    @requests_mock.mock()
    def test_tweets(self, req):
        req.get(base_url + '/account_name/A/all', text=tweets_response, request_headers=request_headers)
        self.assertEqual(tweets, self.storage.get_all_tweets_by_account_name('A'))


if __name__ == '__main__':
    unittest.main()
