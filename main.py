import os
import warnings

from dotenv import load_dotenv
from numba.errors import NumbaPerformanceWarning

from ri_topics.embedder import Embedder
from ri_topics.openreq.ri_storage_twitter import RiStorageTwitter
from ri_topics.router import app
from ri_topics.topics import TopicModelManager
from ri_topics.trend import find_trends

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
    load_dotenv()

    embedder = Embedder()
    rist = RiStorageTwitter(
        base_url=os.getenv('RI_STORAGE_TWITTER_BASE_URL'),
        bearer_token=os.getenv('BEARER_TOKEN'),
    )

    manager = TopicModelManager(embedder, rist)
    model = manager.get('FitbitSupport')

    print(find_trends(model, start='2019-12-01', end='2019-12-03'))

    # app.model_manager = manager
    # app.run(host='0.0.0.0', port='8888')
