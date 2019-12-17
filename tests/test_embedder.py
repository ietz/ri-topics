import unittest
from unittest.mock import Mock

import numpy as np
from sentence_transformers import SentenceTransformer

from ri_topics.embedder import Embedder

EMBEDDING_DIM = 768


class TestEmbedder(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_transformer = Mock(spec=SentenceTransformer, **{
            'encode.side_effect': lambda sents, *args, **kwargs: np.random.random((len(sents), EMBEDDING_DIM))
        })

    def test_embed_texts(self):
        embedder = Embedder(model=self.mock_transformer)
        embeddings = embedder.embed_texts([
            'The text embedding should encode arbitrary texts',
            'and especially also handle multiple sentences! That means it first has to split them.',
            'That is because SBERT handles embedding of single sentences.'
        ])
        self.assertEqual((3, EMBEDDING_DIM), embeddings.shape)


if __name__ == '__main__':
    unittest.main()
