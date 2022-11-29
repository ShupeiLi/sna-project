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
            for j in tqdm(range(10)):
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
            "batch_size": tune.choice([64, 128]),
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
        recall = recall_score(y_true, y_pred, labels=np.unique(y_true), average="macro")
        f1 = f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro")
        return acc, recall, f1


class PubMedGCN(PubMed):
    """
    Implement GCN and our proposed method with pubmed data.
    Args:
        data: Graph data.
        proposed: Use our method. Default: False.
        gnn_epoch: The number of epochs of training GCN model. Default: 10.
        gnn_lr: Learning rate of GCN. Default: 0.01.
        lr: Learning rate of optimizer. Default: 0.001.
        node2vec_epoch: The number of epochs of training node2vec model. Default: 50.
        batch_size: The size of a single batch. Default: 128.
        model_path: Save model. Default: "".
        visual: Prepare for visualization. Default: False.
        Refer to https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html#Node2Vec
        for other parameters.
    """
    class GCN(nn.Module):
        def __init__(self, node_number, embedding_dim, meta_info, proposed):
            super(PubMedGCN.GCN, self).__init__()
            self.embedding = torch.nn.Embedding(num_embeddings=node_number, embedding_dim=embedding_dim)
            self.gcn1 = GCNConv(in_channels=embedding_dim, out_channels=32)
            self.gcn2 = GCNConv(in_channels=32, out_channels=16)
            self.gcn3 = GCNConv(in_channels=16, out_channels=8)
            self.drop1 = torch.nn.Dropout(p=0.1)
            self.drop2 = torch.nn.Dropout(p=0.1)
            if proposed == True:
                self.embedding.from_pretrained(meta_info, freeze=False)
        
        def forward(self, nodes, edges):
            nodes_embedding = self.embedding(nodes)
            nodes_embedding = self.gcn1(nodes_embedding, edges)
            nodes_embedding = torch.relu(nodes_embedding)
            nodes_embedding = self.drop1(nodes_embedding)
            nodes_embedding = self.gcn2(nodes_embedding, edges)
            nodes_embedding = torch.relu(nodes_embedding)
            nodes_embedding = self.drop2(nodes_embedding)
            nodes_embedding = self.gcn3(nodes_embedding, edges)
            nodes_embedding = torch.relu(nodes_embedding)
            return nodes_embedding

    def __init__(self, data, num_nodes, proposed=False, gnn_epoch=10, gnn_lr=0.01, lr=0.001, node2vec_epoch=50, embedding_dim=128, 
            walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128, 
            model_path="", visual=False):
        super().__init__(data, num_nodes, lr, node2vec_epoch, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, p, q, sparse, batch_size, model_path) 
        self.nodes = torch.tensor(range(data.num_nodes))
        self.embedding_dim = embedding_dim
        self.proposed = proposed
        self.gnn_epoch = gnn_epoch
        self.gnn_lr = gnn_lr
        self.visual = visual
        if not proposed:
            self.use_node2vec = False

    def train(self):
        self.model.eval()
        self.clf = PubMedGCN.GCN(node_number=self.num_nodes, embedding_dim=self.embedding_dim, proposed=self.proposed, meta_info=self.model())
        self.clf.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.clf.parameters(), lr=self.gnn_lr)
        if self.visual:
            loss_lst = list()
            acc_lst = list()
            recall_lst = list()
            f1_lst = list()
        for i in range(self.gnn_epoch):
            y_pred = self.clf(self.nodes, self.data.edge_index)
            loss = loss_fn(y_pred[self.data.train_mask], self.data.y[self.data.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.visual:
                loss = loss.detach().cpu().numpy()
                loss_lst.append(loss)
                acc, recall, f1 = self.test()
                acc_lst.append(acc)
                recall_lst.append(recall)
                f1_lst.append(f1)
        if self.visual:
            self.df = pd.DataFrame({"loss": loss_lst, "acc": acc_lst, "recall": recall_lst, "f1": f1_lst})

    def tuning(self):
        def objective(config):
            if self.proposed:
                os.chdir("/home/fangtian/Documents/work/2022Fall/Social Network Analysis for Computer Scientists/project/sna-project/code")
                if os.path.exists(self.model_path):
                    self.model = torch.load(self.model_path)
                else:
                    print("Training node2vec model...")
                    for j in tqdm(range(self.node2vec_epoch)):
                        self.node2vec_train()
            self.model.eval()
            self.clf = PubMedGCN.GCN(node_number=self.num_nodes, embedding_dim=self.embedding_dim, proposed=self.proposed, meta_info=self.model())
            self.clf.train()
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=self.clf.parameters(), lr=config["gnn_lr"])
            for i in range(50):
                y_pred = self.clf(self.nodes, self.data.edge_index)
                loss = loss_fn(y_pred[self.data.train_mask], self.data.y[self.data.train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.clf.eval()
            y_pred = self.clf(self.nodes, self.data.edge_index).detach().cpu().numpy()
            y_pred = y_pred[self.data.val_mask].argmax(1)
            y_true = self.data.y[self.data.val_mask].detach().cpu().numpy()
            acc = accuracy_score(y_true, y_pred)
            tune.report(mean_accuracy=acc)

        search_space = {
            "gnn_lr": tune.grid_search([0.1, 0.01, 0.05, 0.001, 0.005]),
        }

        tuner = tune.Tuner( 
                objective, 
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    num_samples=5,
                    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max")),
                    run_config=air.RunConfig(local_dir="../results", name="pubmed_our_tuning")
                )
        results = tuner.fit()

    @torch.no_grad()
    def test(self):
        self.clf.eval()
        y_pred = self.clf(self.nodes, self.data.edge_index).detach().cpu().numpy()
        y_pred = y_pred[self.data.test_mask].argmax(1)
        y_true = self.data.y[self.data.test_mask].detach().cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, labels=np.unique(y_true), average="macro")
        f1 = f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro")
        return acc, recall, f1


def visualization(gcn_model, our_model):
    def visual_train(model):
        model.model = torch.load(model.model_path)
        model.visual = True
        model.train()
        model.visual = False
        df = model.df
        print(f"acc: {df.iloc[-1, 1]}")
        print(f"recall: {df.iloc[-1, 2]}")
        print(f"f1: {df.iloc[-1, 3]}")
        return df
    gcn_df = visual_train(gcn_model)
    our_df = visual_train(our_model)
    title = ["Train Loss", "Test Acc", "Test Recall", "Test F1"]
    time.sleep(5)
    for i in range(4):
        plt.plot(list(range(1, 51)), gcn_df.iloc[:, i].tolist(), label="GCN")
        plt.plot(list(range(1, 51)), our_df.iloc[:, i].tolist(), label="Proposed")
        plt.title(title[i])
        plt.legend()
        plt.show()


if __name__ == "__main__":
    pubmed_model = PubMed(pubmed_data, node_arr.shape[0], node2vec_epoch=10, lr=0.01, embedding_dim=128, walk_length=20, p=4, q=4, batch_size=64, model_path="../results/models/pubmed.pt")
    pubmed_model.node2vec_save()
    pubmed_model.tuning()
    pubmed_model.main()
    pubmed_gcn_model = PubMedGCN(pubmed_data, node_arr.shape[0], node2vec_epoch=10, lr=0.01, embedding_dim=128, walk_length=20, p=4, q=4, batch_size=64, gnn_epoch=50, gnn_lr=0.05, proposed=False, model_path="../results/models/pubmed.pt")
    pubmed_gcn_model.tuning()
    pubmed_gcn_model.main()
    pubmed_our_model = PubMedGCN(pubmed_data, node_arr.shape[0], node2vec_epoch=10, lr=0.01, embedding_dim=128, walk_length=20, p=4, q=4, batch_size=64, gnn_epoch=50, gnn_lr=0.1, proposed=True, model_path="../results/models/pubmed.pt")
    pubmed_our_model.tuning()
    pubmed_our_model.main()
    visualization(pubmed_gcn_model, pubmed_our_model)
