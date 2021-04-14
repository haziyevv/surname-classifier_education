from surname_classification.data.vocabulary import Vocabulary
import pandas as pd
import numpy as np


class SurnameVectorizer(object):
    def __init__(self, surname_vocab: Vocabulary, nation_vocab: Vocabulary):
        self.surname_vocab = surname_vocab
        self.nation_vocab = nation_vocab

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """
        Args:
            df:
        Returns:
        """
        surname_vocab = Vocabulary()
        nation_vocab = Vocabulary()

        for _, row in df.iterrows():
            surname = row["surname"]
            nationality = row["nationality"]
            nation_vocab.add_token(nationality)

            for char in surname:
                surname_vocab.add_token(char)
        return cls(surname_vocab, nation_vocab)

    def vectorize(self, surname: str) -> np.array:
        """
        Args:
            surname:
        Returns: collapsed vectorized form of the surname
        """
        vec = np.zeros(len(self.surname_vocab), dtype=np.float32)
        for char in surname:
            vec[self.surname_vocab.token_to_idx[char]] = 1

        return vec
