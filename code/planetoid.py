# -*- coding: utf-8 -*-

from base import *
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score


pubmed_data = Planetoid(root="../data/", name="PubMed")
pubmed_data = pubmed_data[0]
edge_arr = np.transpose(pubmed_data.edge_index.detach().cpu().numpy())
node_arr = np.unique(edge_arr.flatten())


class PubMed(BaseModel):
    """
    Implement algorithms with pubmed data.
    Args:
        data: Graph data.
        num_nodes: Number of nodes in the graph.
        lr: Learning rate of optimizer. Default: 0.001.
        node2vec_epoch: The number of epochs of training node2vec model. Default: 50.
        batch_size: The size of a single batch. Default: 128.
        Refer to https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html#Node2Vec
        for other parameters.
    """
    def __init__(self, data, num_nodes, lr=0.001, node2vec_epoch=50, embedding_dim=128, walk_length=20, context_size=10,
            walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128):
        self.use_node2vec = True
        self.data = data
        self.node2vec_epoch = node2vec_epoch
        self.preprocessing(num_nodes, lr, embedding_dim, walk_length, context_size, walks_per_node, 
                num_negative_samples, p, q, sparse, batch_size)

    def train(self):
        self.model.eval()
        self.clf = LogisticRegression()
        self.clf.fit(self.model()[self.data.train_mask].detach().cpu().numpy(), self.data.y[self.data.train_mask].detach().cpu().numpy())

    @torch.no_grad()
    def test(self):
        y_pred = self.clf.predict(self.model()[self.data.test_mask].detach().cpu().numpy())
        y_true = self.data.y[self.data.test_mask].detach().cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, labels=np.unique(y_true), average="weighted")
        f1 = f1_score(y_true, y_pred, labels=np.unique(y_true), average="weighted")
        return acc, recall, f1


if __name__ == "__main__":
    pubmed_model = PubMed(pubmed_data, node_arr.shape[0], node2vec_epoch=5)
    pubmed_model.main()
