import os
from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ri_topics.preprocessing import Document


class Embedder:
    def embed(self, docs: List[Document]):
        raise NotImplementedError()

    def embed_texts(self, texts: List[str], show_progess=True) -> np.ndarray:
        logger.info('Preprocessing texts')
        text_it = texts if not show_progess else tqdm(texts, desc='Preprocessing', unit='Tweets')
        docs = [Document(text) for text in text_it]

        self.embed(docs)
        return np.array([doc.embedding for doc in docs])


class SbertEmbedder(Embedder):
    def __init__(self):
        self.model = SentenceTransformer(os.getenv('SBERT_MODEL'))

    def embed(self, docs: List[Document]):
        sentences = list(sent for doc in docs for sent in doc.sentences)
        logger.info(f'Generating embeddings for {len(sentences)} sentences')
        print('Generating embeddings for sentences', [str(sent) for sent in sentences])
        embeddings = self.model.encode([str(sent) for sent in sentences], show_progress_bar=True)
        for sentence, embedding in zip(sentences, embeddings):
            sentence.embedding = embedding
