import logging
import os

import numpy as np
from dotenv import load_dotenv

from ri_topics.clustering import Clusterer
from ri_topics.embedder import Embedder
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter
from ri_topics.preprocessing import Document
from ri_topics.summary import Summarizer

if __name__ == '__main__':
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    rist = RiStorageTwitter(
        base_url=os.getenv('RI_STORAGE_TWITTER_BASE_URL'),
        bearer_token=os.getenv('BEARER_TOKEN'),
    )

    tweets = rist.get_all_tweets_by_account_name('FitbitSupport')
    texts = [tweet.text for tweet in tweets]
    docs = [Document(text) for text in texts]
    embedder = Embedder()
    embedder.embed(docs)
    embeddings = np.array([doc.embedding for doc in docs])

    clusterer = Clusterer()
    assignment = clusterer.fit(embeddings, n_components=10, n_neighbors=40, min_dist=0, min_cluster_size=30, min_samples=20)

    summarizer = Summarizer()
    summarizer.fit(docs, assignment)

    print(summarizer._representatives)

    logger.info('Starting server')
    app.run(host='0.0.0.0', port='8888')
