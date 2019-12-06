from dataclasses import dataclass
from typing import List

import numpy as np
from spacy.lang.en import English


class Sentencizer:
    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def split(self, text: str) -> List[str]:
        return [str(sent) for sent in self.nlp(text).sents]


class Document:
    sentencizer = Sentencizer()

    def __init__(self, text):
        self.sentences = [Sentence(sent) for sent in self.sentencizer.split(text)]

    @property
    def embedding(self) -> np.ndarray:
        return np.array([sent.embedding for sent in self.sentences if sent.embedding is not None]) \
                 .mean(axis=0)

    def __str__(self):
        return ' '.join([str(sent) for sent in self.sentences])


@dataclass
class Sentence:
    text: str
    embedding: np.ndarray = None

    def __str__(self):
        return self.text
