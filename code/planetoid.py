# -*- coding: utf-8 -*-

from base import *
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
import pandas as pd
import matplotlib.pyplot as plt
import time


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
        model_path: Save model. Default: "".
        Refer to https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html#Node2Vec
        for other parameters.
    """
    def __init__(self, data, num_nodes, lr=0.001, node2vec_epoch=50, embedding_dim=128, walk_length=20, context_size=10,
            walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128, model_path=""):
        self.use_node2vec = True
        self.data = data
        self.num_nodes = num_nodes
        self.node2vec_epoch = node2vec_epoch
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.sparse = sparse
        self.model_path = model_path
        self.preprocessing(num_nodes, lr, embedding_dim, walk_length, context_size, walks_per_node, 
                num_negative_samples, p, q, sparse, batch_size)

    def train(self):
        self.model.eval()
        self.clf = LogisticRegression()
        self.clf.fit(self.model()[self.data.train_mask].detach().cpu().numpy(), self.data.y[self.data.train_mask].detach().cpu().numpy())

    def tuning(self):
        def objective(config):
            self.preprocessing(self.num_nodes, config["lr"], config["embedding_dim"], config["walk_length"], 
                    self.context_size, self.walks_per_node, self.num_negative_samples, config["p"], config["q"], 
                    self.sparse, config["batch_size"])
            for j in tqdm(range(config["node2vec_epoch"])):
                self.node2vec_train()
            self.train()
            y_pred = self.clf.predict(self.model()[self.data.val_mask].detach().cpu().numpy())
            y_true = self.data.y[self.data.val_mask].detach().cpu().numpy()
            acc = accuracy_score(y_true, y_pred)
            tune.report(mean_accuracy=acc)

        search_space = {
            "lr": tune.grid_search([0.001, 0.01, 0.1]),
            "embedding_dim": tune.choice([64, 128, 256]),
            "walk_length": tune.choice([10, 20, 30]),
            "p": tune.choice([0.25, 0.5, 1, 2, 4]),
            "q": tune.choice([0.25, 0.5, 1, 2, 4]),
            "batch_size": tune.choice([64, 128, 256]),
            "node2vec_epoch": tune.choice([10, 20])
        }

        tuner = tune.Tuner( 
                objective, 
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    num_samples=20,
                    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max")),
                run_config=air.RunConfig(local_dir="../results", name="pubmed_node2vec_tuning")
                )
        results = tuner.fit()

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
    pubmed_model.tuning()
#   pubmed_model.main()
