import torch.nn as nn
import torch.nn.functional as F


class SurnameClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(SurnameClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        Args:
            x:
        Returns:
        """
        out = F.relu(self.fc1(x))
        out = self.fc2(out)

        return out
