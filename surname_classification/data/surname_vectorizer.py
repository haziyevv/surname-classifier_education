import numpy as np
from surname_classification.data.vocabulary import Vocabulary


class SurnameVectorizer(object):
    def __init__(self, surname_vocab, nation_vocab):
        """
        Takes the surname and converts it to one hot representation
        Args:
            surname_vocab: surname vocabulary. Includes 32 characters as this is a character level vocab.
            nation_vocab: vocabulary for all the different nations in the dataset.
        """

        self.surname_vocab = surname_vocab
        self.nation_vocab = nation_vocab

    def vectorize(self, surname):
        """
        Takes the surname and converts it to one hot representation
        Args:
            surname:

        Returns:
            One hot vector, with all zeros but indices of the characters in the surname as 1
        """
        one_hot = np.zeros((len(self.surname_vocab)), dtype=np.float32)
        for token in surname:
            one_hot[self.surname_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, surname_df):
        """
        Args:
            surname_df:
        Returns:
        """
        surname_vocab = Vocabulary()
        nationality_vocab = Vocabulary()

        for index, row in surname_df.iterrows():
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)
        return cls(surname_vocab, nationality_vocab)
