# -*- coding: utf-8 -*-

import torch
from torch_geometric.datasets import PPI
from torch_geometric.nn import Node2Vec

import sys
import numpy as np

data_path = "../data"

train_dataset = PPI(root=data_path)
val_dataset = PPI(root=data_path, split="val")
test_dataset = PPI(root=data_path, split="test")
print(f"Train data: {train_dataset}.")
print(f"Val data: {val_dataset}.")
print(f"Test data: {test_dataset}.")


class BaseModel():
    def __init__(self, data, test_data):
        self.data = data
        self.test_data = test_data
        self.preprocessing()

    def preprocessing(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Node2Vec(self.data.edge_index, embedding_dim=128, walk_length=20,
                         context_size=10, walks_per_node=10,
                         num_negative_samples=1, p=1, q=1, sparse=True).to(self.device)

        num_workers = 0 if sys.platform.startswith('win') else 4
        self.loader = self.model.loader(batch_size=128, shuffle=True,
                              num_workers=num_workers)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

    def train(self):
        self.model.train()
        total_loss = 0
        for pos_rw, neg_rw in self.loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        z = self.model()
        acc = self.model.test(self.data.x, self.data.y,
                              self.test_data.x, self.test_data.y,
                              max_iter=150)
        return acc


if __name__ == "__main__":
    graph_model = BaseModel(val_dataset[0], test_dataset[0])
    for epoch in range(1, 11):
        loss = graph_model.train()
#       acc = graph_model.test()
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")#, Acc: {acc:.4f}")
    print(graph_model.model())
    print(graph_model.model().shape)
