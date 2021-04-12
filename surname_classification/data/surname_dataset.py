import pandas as pd
from torch.utils.data import Dataset
from surname_classification.data.surname_vectorizer import SurnameVectorizer
import numpy as np
import collections
import pdb

class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        """
        Args:
            surname_df:
            vectorizer:
        """

        self.surname_df = surname_df
        self._vectorizer = vectorizer

        self.train_df = self.surname_df[self.surname_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.surname_df[self.surname_df.split == "val"]
        self.val_size = len(self.val_df)

        self.test_df = self.surname_df[self.surname_df.split == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {"train": (self.train_df, self.train_size),
                             "val": (self.val_df, self.val_size),
                             "test": (self.test_df, self.test_size)}

        self.set_split("train")

    def get_vectorizer(self):
        return self._vectorizer

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv) -> object:
        """
        Args:
            surname_csv:

        Returns:
        """
        surname_df = SurnameDataset.create_clean_df(surname_csv)
        return cls(surname_df, SurnameVectorizer.from_dataframe(surname_df))

    def set_split(self, split="train"):
        """
        Args:
            split:
        Returns:
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    @staticmethod
    def create_clean_df(filename):
        seed = 1
        final_list = []
        np.random.seed(seed)
        train_proportion = 0.8
        val_proportion = 0.1

        df = pd.read_csv(filename)
        by_rating = collections.defaultdict(list)
        for x, row in df.iterrows():
            by_rating[row["nationality"]].append(row.to_dict())

        for _, item_list in sorted(by_rating.items()):
            np.random.shuffle(item_list)
            n_total = len(item_list)

            n_train = int(train_proportion * n_total)
            n_val = int(val_proportion * n_total)
            n_test = n_total - n_train - n_val

            # Give data point a split attribute
            for item in item_list[:n_train]:
                item['split'] = 'train'
            for item in item_list[n_train:n_train + n_val]:
                item['split'] = 'val'
            for item in item_list[n_train + n_val:n_train + n_val + n_test]:
                item['split'] = 'test'
            # Add to final list
            final_list.extend(item_list)
        final_reviews = pd.DataFrame(final_list)

        return final_reviews

    def __getitem__(self, index):
        """
        Args:
            index:
        Returns:
        """
        row = self._target_df.iloc[index]

        surname_vector = self._vectorizer.vectorize(row.surname)
        nation_index = self._vectorizer.nation_vocab.lookup_token(row.nationality)

        return {"x_data": surname_vector,
                "y_target": nation_index}
