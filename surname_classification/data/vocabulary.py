class Vocabulary(object):
    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}

    def add_token(self, token):
        """
        Args:
            token:
        Returns: index of the token in token_to_idex
        """
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def __len__(self):
        return len(self.token_to_idx)
