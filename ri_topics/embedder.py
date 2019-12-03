import logging
import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ri_topics.models import Tweet
from ri_topics.preprocessing import Document

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(os.getenv('SBERT_MODEL'))

    def embed(self, docs: List[Document]):
        sentences = list(sent for doc in docs for sent in doc.sentences)
        logger.info(f'Generating embeddings for {len(sentences)} sentences')
        embeddings = self.model.encode([str(sent) for sent in sentences])
        for sentence, embedding in zip(sentences, embeddings):
            sentence.embedding = embedding

    def embed_tweets(self, tweets: List[Tweet], show_progess=True) -> np.ndarray:
        texts = [tweet.text for tweet in tweets]
        text_it = texts if not show_progess else tqdm(texts, desc='Preprocessing', unit='Tweets')
        docs = [Document(text) for text in text_it]

        self.embed(docs)
        return np.array([doc.embedding for doc in docs])
