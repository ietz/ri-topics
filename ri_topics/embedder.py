import os
from typing import List

from sentence_transformers import SentenceTransformer

from ri_topics.preprocessing import Document


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(os.getenv('SBERT_MODEL'))

    def embed(self, docs: List[Document]):
        sentences = list(sent for doc in docs for sent in doc.sentences)
        embeddings = self.model.encode([str(sent) for sent in sentences])
        for sentence, embedding in zip(sentences, embeddings):
            sentence.embedding = embedding
