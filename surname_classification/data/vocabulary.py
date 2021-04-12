
class Vocabulary:
    def __init__(self):
        self.token_to_idx = dict()
        self.idx_to_token = dict()

    def add_token(self, token):
        """
        To add tokens to the given vocabulary
        Args:
            token:
        Returns:
            The id of the added token
        """
        if token in self.token_to_idx:
            id_ = self.token_to_idx[token]
        else:
            id_ = len(self.token_to_idx)
            self.token_to_idx[token] = id_
            self.idx_to_token[id_] = token
        return id_

    def lookup_token(self, token):
        """
        Returns the id of the input token token
        Args:
            token:
        Returns:
            id of the token
        """
        return  self.token_to_idx[token]

    def __len__(self):
        return len(self.token_to_idx)
