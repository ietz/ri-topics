import logging
import os

from dotenv import load_dotenv

from ri_topics.embedder import Embedder
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter
from ri_topics.topics import TopicModel

if __name__ == '__main__':
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    embedder = Embedder()
    rist = RiStorageTwitter(
        base_url=os.getenv('RI_STORAGE_TWITTER_BASE_URL'),
        bearer_token=os.getenv('BEARER_TOKEN'),
    )

    model = TopicModel('FitbitSupport', embedder=embedder, storage=rist)
    model.train(n_components=10, n_neighbors=40, min_dist=0, min_cluster_size=30, min_samples=20)

    # logger.info('Starting server')
    # app.run(host='0.0.0.0', port='8888')
