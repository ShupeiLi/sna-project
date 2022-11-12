# -*- coding: utf-8 -*-

import abc
import sys
from functools import wraps
from time import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.nn import Node2Vec, GCNConv


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
    Implement node2vec algorithm and define the workflow.
    """
    def preprocessing(self, num_nodes, lr=0.001, embedding_dim=128, walk_length=20, context_size=10,
            walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Node2Vec(self.data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length, 
                context_size=context_size, walks_per_node=walks_per_node, num_negative_samples=num_negative_samples, 
                num_nodes=num_nodes, p=p, q=q, sparse=sparse).to(self.device)

        num_workers = 0 if sys.platform.startswith("win") else 4
        self.loader = self.model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=lr)

    def node2vec_train(self):
        self.model.train()
        for pos_rw, neg_rw in self.loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()

    @abc.abstractmethod
    def train(self):
        """
        Train the model.
        """

    @abc.abstractmethod
    def test(self):
        """
        Evaluate model performance.
        """

    @cal_time
    def main(self):
        if self.use_node2vec == True:
            print("Training node2vec model...")
            for epoch in tqdm(range(self.node2vec_epoch)):
                self.node2vec_train()
        print("Training model for the task...")
        self.train()
        print("Evaluating model...")
        acc, recall, f1 = self.test()
        print(f"Test Acc: {acc:.4f}, Test Recall: {recall:.4f}, Test F1: {f1:.4f}.")


class GCN(nn.Module):
    def __init__(self, node_number, embedding_dim, meta_info, proposed):
        super(GCN, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=node_number, embedding_dim=embedding_dim)
        self.gcn = GCNConv(in_channels=embedding_dim, out_channels=embedding_dim)
        if proposed == True:
            self.embedding.from_pretrained(meta_info, freeze=False)
    
    def forward(self, nodes, edges):
        nodes_embedding = self.embedding(nodes)
        nodes_embedding = self.gcn(nodes_embedding,edges)
        nodes_embedding = torch.relu(nodes_embedding)
        return nodes_embedding
