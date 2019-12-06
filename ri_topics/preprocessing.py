from dataclasses import dataclass

import numpy as np
from spacy.lang.en import English


class Document:
    def __init__(self, text):
        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        doc = nlp(text)
        self.sentences = [Sentence(str(sent)) for sent in doc.sents]

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
