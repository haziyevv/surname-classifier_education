import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from argparse import Namespace
from surname_classification.data.surname_dataset import SurnameDataset
from surname_classification.models.surname_classifier import  SurnameClassifier
import pdb


def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to("cpu")
        yield out_data_dict

args = Namespace(
    surname_csv="../../data/raw/surnames.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="model_storage/ch4/surname_mlp",
    # Model hyperparameters
    hidden_dim=300,
    # Training hyperparameters,
    seed=1337,
    num_epochs=100,
    early_stopping_criteria=5,
    learning_rate=0.001,
    batch_size=64
)
dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
vectorizer = dataset.get_vectorizer()
classifier = SurnameClassifier(input_dim=len(vectorizer.surname_vocab),
                               hidden_dim=args.hidden_dim,
                               output_dim=len(vectorizer.nation_vocab))

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

for epoch_index in range(args.num_epochs):
    dataset.set_split("train")
    batch_generator = generate_batches(dataset, args.batch_size)
    running_loss = 0.0
    running_acc = 0.0
    classifier.train()

    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        y_pred = classifier(batch_dict["x_data"])
        loss = loss_func(y_pred, batch_dict["y_target"])
        loss_batch = loss.item()
        loss.backward()
        optimizer.step()
        if batch_index % 50 == 0:
            print(epoch_index, batch_index, loss.item() / 128)

