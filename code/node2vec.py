# -*- coding: utf-8 -*-

import abc
import sys
from functools import wraps
from time import time
import torch
from torch_geometric.nn import Node2Vec


def cal_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"Time: func {func.__name__} took {end_time - start_time:2.4f} s.")
        return result
    return wrapper


class BaseModel(abc.ABC):
    """
    Node2vec algorithm implementation.
    """
    def preprocessing(self, num_nodes, lr=0.001, embedding_dim=128, walk_length=20, context_size=10,
            walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True,
            batch_size=128):
        print("Preprocessing...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Node2Vec(self.data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length, 
                context_size=context_size, walks_per_node=walks_per_node, num_negative_samples=num_negative_samples, 
                num_nodes=num_nodes, p=p, q=q, sparse=sparse).to(self.device)

        num_workers = 0 if sys.platform.startswith("win") else 4
        self.loader = self.model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=lr)
        print("Done.")

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

    @abc.abstractmethod
    def test(self):
        """
        Evaluate model performance.
        """

    @cal_time
    def main(self):
        if not hasattr(self, "epochs"):
            raise AttributeError(f"Class attribute 'epochs' is not defined.")
        for epoch in range(1, self.epochs + 1):
            loss = self.train()
            acc, recall, f1 = self.test()
            print(f"Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test Recall: {recall:.4f}, Test F1: {f1:.4f}.")
