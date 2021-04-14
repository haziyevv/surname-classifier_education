from surname_classification.data.surname_dataset import SurnameDataset
from surname_classifier import SurnameClassifier
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import Namespace
import pdb

args = Namespace(
    input_dim=84,
    hidden_dim=100,
    learning_rate=0.001,
    surname_csv="../../data/surnames.csv",
    batch_size=64,
    num_epochs=5,
    shuffle=True,
    seed=1337
)

dataset = SurnameDataset.load_dataframe_and_make_vectorizer(args.surname_csv)

vectorizer = dataset.get_vectorizer()
train_loader = DataLoader(batch_size=args.batch_size,
                          shuffle=args.shuffle,
                          dataset=dataset)


model = SurnameClassifier(input_dim=args.input_dim,
                          hidden_dim=args.hidden_dim,
                          out_dim=len(vectorizer.nation_vocab))

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(args.num_epochs):
    loss_train = 0.0
    for xs, ys in train_loader:
        outputs = model(xs)

        loss = loss_fn(outputs, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
    print(f"Epoch {epoch} loss {loss_train/len(train_loader)}")

