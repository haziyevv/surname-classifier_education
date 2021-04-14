from surname_classification.data.surname_vectorizer import SurnameVectorizer
from torch.utils.data import Dataset
import pandas as pd
import random
from collections import defaultdict


class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        self.surname_df = surname_df
        self._vectorizer = vectorizer

        self.train_df = self.surname_df[self.surname_df == "train"]
        self.train_size = len(self.train_df)

        self.dev_df = self.surname_df[self.surname_df == "val"]
        self.dev_size = len(self.dev_df)

        self.test_df = self.surname_df[self.surname_df == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {"train": (self.train_df, self.train_size),
                             "dev": (self.dev_df, self.dev_size),
                             "test": (self.test_df, self.test_size)}
        self.set_split("train")

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, target="train"):
        self.target = target
        self.target_df, self.target_size = self._lookup_dict[self.target]

    def __getitem__(self, index):
        row = self.surname_df.loc[index]
        surname_vector = self._vectorizer.vectorize(row["surname"])
        nation = self._vectorizer.nation_vocab.token_to_idx[row["nationality"]]

        return surname_vector, nation

    def __len__(self):
        return len(self.target_df)

    @classmethod
    def load_dataframe_and_make_vectorizer(cls, csv_file):
        df = create_clean_df(csv_file)
        return cls(df, SurnameVectorizer.from_dataframe(df))


def create_clean_df(csv_file):
    """
    Args:
        csv_file:
    Returns: cleaned dataframe
    """
    df = pd.read_csv("../../data/raw/surnames.csv")

    nation_dict = defaultdict(list)
    for _, row in df.iterrows():
        nation_dict[row["nationality"]].append(row)

    final_list = []

    for _, item_list in nation_dict.items():
        train_num = int(len(item_list) * 0.8)
        val_num = train_num + int(len(item_list) * 0.1)

        random.shuffle(item_list)
        for i, item in enumerate(item_list):
            if i < train_num:
                item["split"] = "train"
            elif train_num < i < val_num:
                item["split"] = "dev"
            else:
                item["split"] = "test"
            final_list.append(item)

    return pd.DataFrame(final_list)
