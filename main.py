import os
from dotenv import load_dotenv

from ri_topics.embedder import Embedder
from ri_topics.logging import setup_logging
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter
from ri_topics.router import app
from ri_topics.topics import TopicModelManager

if __name__ == '__main__':
    setup_logging()
    load_dotenv()

    embedder = Embedder()
    rist = RiStorageTwitter(
        base_url=os.getenv('RI_STORAGE_TWITTER_BASE_URL'),
        bearer_token=os.getenv('BEARER_TOKEN'),
    )

    manager = TopicModelManager(embedder, rist)
    manager.prepare_all()

    app.model_manager = manager
    app.run(host='0.0.0.0', port='8888')
