# -*- coding: utf-8 -*-

from base import *
import torch
import numpy as np
from torch_geometric.datasets import Entities
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
import pandas as pd
import matplotlib.pyplot as plt
import time


aifb_data = Entities(root="../data/Entities/AIFB/", name="AIFB")
aifb_data = aifb_data[0]
mutag_data = Entities(root="../data/Entities/MUTAG/", name="MUTAG")
mutag_data = mutag_data[0]


class EntitiesBase(BaseModel):
    """
    Implement node2vec with entities data.
    Args:
        data: Graph data.
        multi: Multiclass classification or binary class classification. Default: True.
        lr: Learning rate of optimizer. Default: 0.001.
        node2vec_epoch: The number of epochs of training node2vec model. Default: 50.
        batch_size: The size of a single batch. Default: 128.
        model_path: Save model. Default: "".
        Refer to https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html#Node2Vec
        for other parameters.
    """
    def __init__(self, data, multi=True, lr=0.001, node2vec_epoch=50, embedding_dim=128, walk_length=20, context_size=10,
            walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128, model_path=""):
        self.use_node2vec = True
        self.data = data
        self.multi = multi
        self.node2vec_epoch = node2vec_epoch
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.sparse = sparse
        self.model_path = model_path
        self.preprocessing(data.num_nodes, lr, embedding_dim, walk_length, context_size, walks_per_node, 
                num_negative_samples, p, q, sparse, batch_size)

    def train(self):
        self.model.eval()
        self.clf = LogisticRegression()
        self.clf.fit(self.model()[self.data.train_idx].detach().cpu().numpy(), self.data.train_y.detach().cpu().numpy())

    def tuning(self):
        def objective(config):
            self.preprocessing(self.data.num_nodes, config["lr"], config["embedding_dim"], config["walk_length"], 
                    self.context_size, self.walks_per_node, self.num_negative_samples, config["p"], config["q"], 
                    self.sparse, config["batch_size"])
            for j in tqdm(range(config["node2vec_epoch"])):
                self.node2vec_train()
            self.train()
            acc, recall, f1 = self.test()
            tune.report(mean_accuracy=acc)

        search_space = {
            "lr": tune.grid_search([0.001, 0.01, 0.1]),
            "embedding_dim": tune.choice([64, 128, 256]),
            "walk_length": tune.choice([10, 20, 30]),
            "p": tune.choice([0.25, 0.5, 1, 2, 4]),
            "q": tune.choice([0.25, 0.5, 1, 2, 4]),
            "batch_size": tune.choice([64, 128, 256]),
            "node2vec_epoch": tune.choice([10, 20, 30, 40, 50, 60])
        }

        tuner = tune.Tuner( 
                objective, 
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    num_samples=20,
                    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max")),
                run_config=air.RunConfig(local_dir="../results", name="node2vec_tuning")
                )
        results = tuner.fit()
   
    @torch.no_grad()   
    def test(self):
        y_pred = self.clf.predict(self.model()[self.data.test_idx].detach().cpu().numpy())
        y_true = self.data.test_y.detach().cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        if self.multi:
            recall = recall_score(y_true, y_pred, labels=np.unique(y_true), average="weighted")
            f1 = f1_score(y_true, y_pred, labels=np.unique(y_true), average="weighted")
        else:
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        return acc, recall, f1
        

class EntitiesGCN(EntitiesBase):
    """
    Implement GCN and our proposed method with entities data.
    Args:
        data: Graph data.
        proposed: Use our method. Default: False.
        gnn_epoch: The number of epochs of training GCN model. Default: 10.
        gnn_lr: Learning rate of GCN. Default: 0.01.
        multi: Multiclass classification or binary class classification. Default: True.
        lr: Learning rate of optimizer. Default: 0.001.
        node2vec_epoch: The number of epochs of training node2vec model. Default: 50.
        batch_size: The size of a single batch. Default: 128.
        model_path: Save model. Default: "".
        visual: Prepare for visualization. Default: False.
        Refer to https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html#Node2Vec
        for other parameters.
    """
    def __init__(self, data, proposed=False, gnn_epoch=10, gnn_lr=0.01, multi=True, lr=0.001, node2vec_epoch=50, embedding_dim=128, 
            walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128, 
            model_path="", visual=False):
        super().__init__(data, multi, lr, node2vec_epoch, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, p, q, sparse, batch_size, model_path) 
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
        self.clf = GCN(node_number=self.data.num_nodes, embedding_dim=self.embedding_dim, proposed=self.proposed, meta_info=self.model())
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
            loss = loss_fn(y_pred[self.data.train_idx],self.data.train_y)
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
            self.clf = GCN(node_number=self.data.num_nodes, embedding_dim=self.embedding_dim, proposed=self.proposed, meta_info=self.model())
            self.clf.train()
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=self.clf.parameters(), lr=config["gnn_lr"])
            for i in range(config["gnn_epoch"]):
                y_pred = self.clf(self.nodes, self.data.edge_index)
                loss = loss_fn(y_pred[self.data.train_idx],self.data.train_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            acc, recall, f1 = self.test()
            tune.report(mean_accuracy=acc)

        search_space = {
            "gnn_lr": tune.grid_search([0.1, 0.01, 0.05]),
            "gnn_epoch": tune.choice([20, 30, 40, 50, 60])
        }

        tuner = tune.Tuner( 
                objective, 
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    num_samples=20,
                    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max")),
                run_config=air.RunConfig(local_dir="../results", name="gcn_tuning")
                )
        results = tuner.fit()

    @torch.no_grad()
    def test(self):
        self.clf.eval()
        y_pred = self.clf(self.nodes, self.data.edge_index).detach().cpu().numpy()
        y_pred = y_pred[self.data.test_idx].argmax(1)
        y_true = self.data.test_y.detach().cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        if not self.multi:
            y_true_one_hot = np.zeros((y_true.size, y_true.max() + 1))
            y_true_one_hot[np.arange(y_true.size), y_true] = 1
        recall = recall_score(y_true, y_pred, labels=np.unique(y_true), average="weighted")
        f1 = f1_score(y_true, y_pred, labels=np.unique(y_true), average="weighted")
        return acc, recall, f1

def visualization(gcn_model, our_model):
    def visual_train(model):
        model.model = torch.load(model.model_path)
        model.visual = True
        model.train()
        model.visual = False
        df = model.df
        print(f"acc: {max(df.iloc[:, 1].tolist())}")
        print(f"recall: {max(df.iloc[:, 2].tolist())}")
        print(f"f1: {max(df.iloc[:, 3].tolist())}")
        return df
    gcn_df = visual_train(gcn_model)
    our_df = visual_train(our_model)
    title = ["Train Loss", "Test Acc", "Test Recall", "Test F1"]
    time.sleep(5)
    for i in range(4):
        plt.plot(list(range(1, 101)), gcn_df.iloc[:, i].tolist(), label="GCN")
        plt.plot(list(range(1, 101)), our_df.iloc[:, i].tolist(), label="Proposed")
        plt.title(title[i])
        plt.legend()
        plt.show()


if __name__ == "__main__":
#   aifb_model = EntitiesBase(aifb_data, node2vec_epoch=10, lr=0.1, embedding_dim=128, walk_length=30, p=0.25, q=0.25, batch_size=128, model_path="../results/models/aifb.pt")
#   aifb_model.node2vec_save()
#   aifb_model.tuning()
#   aifb_model.main()
#   aifb_gcn_model = EntitiesGCN(aifb_data, node2vec_epoch=10, lr=0.1, embedding_dim=128, walk_length=30, p=0.25, q=0.25, batch_size=128, gnn_epoch=100, gnn_lr=0.01, proposed=False, model_path="../results/models/aifb.pt")
#   aifb_gcn_model.tuning()
#   aifb_gcn_model.main()
#   aifb_our_model = EntitiesGCN(aifb_data, node2vec_epoch=10, lr=0.1, embedding_dim=128, walk_length=30, p=0.25, q=0.25, batch_size=128, gnn_epoch=100, gnn_lr=0.01, proposed=True, model_path="../results/models/aifb.pt")
#   aifb_our_model.tuning()
#   aifb_our_model.main()
#   visualization(aifb_gcn_model, aifb_our_model)

    mutag_model = EntitiesBase(mutag_data, multi=False, node2vec_epoch=20, lr=0.01, embedding_dim=128, walk_length=30, p=2, q=4, batch_size=64, model_path="../results/models/mutag.pt")
    mutag_model.node2vec_save()
#   mutag_model.tuning()
#   mutag_model.main()
#   mutag_gcn_model = EntitiesGCN(mutag_data, multi=False, node2vec_epoch=50, lr=0.01, embedding_dim=128, walk_length=20, p=0.5, q=0.25, batch_size=128, gnn_epoch=100, gnn_lr=0.01, proposed=False, model_path="../results/models/mutag.pt")
#   mutag_gcn_model.tuning()
#   mutag_gcn_model.main()
#   mutag_our_model = EntitiesGCN(mutag_data, multi=False, node2vec_epoch=50, lr=0.01, embedding_dim=128, walk_length=20, p=0.5, q=0.25, batch_size=128, gnn_epoch=100, gnn_lr=0.01, proposed=True, model_path="../results/models/mutag.pt")
#   mutag_gcn_model.tuning()
#   mutag_gcn_model.main()
#   visualization(mutag_gcn_model, mutag_our_model)
