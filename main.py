import logging
import os
from dotenv import load_dotenv

from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter
from ri_topics.router import app


if __name__ == '__main__':
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    rist = RiStorageTwitter(
        base_url=os.getenv('RI_STORAGE_TWITTER_BASE_URL'),
        bearer_token=os.getenv('BEARER_TOKEN'),
    )

    logger.info('Fetching account names')
    accounts = rist.get_all_account_names()
    logger.info(f'Got {len(accounts)} accounts: {accounts}')
    logger.info(f'Fetching tweets by {accounts[0]}')
    tweets = rist.get_all_tweets_by_account_name(accounts[0])
    logger.info(f'Got {len(tweets)} tweets')
    logger.info(f'First tweet:')
    logger.info(tweets[0])

    logger.info('Starting server')
    app.run(host='0.0.0.0', port='8888')
