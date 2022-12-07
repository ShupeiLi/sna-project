# -*- coding: utf-8 -*-

import pickle
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from base import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from torch_geometric.nn import GCNConv
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler


def not_connected_store(path, edge_arr, node_arr):
    print("Creating a dictionary of non-existed edges...")
    G = nx.MultiGraph()
    G.add_edges_from(edge_arr)

    not_connected_dict = dict()
    for index in tqdm(range(node_arr.shape[0])):
        node = node_arr[index]
        records = list()
        shortest_path = nx.shortest_path_length(G, source=node)
        for item in shortest_path.items():
            if item[1] > 1:
                records.append(item[0])
        not_connected_dict[node] = records

    with open(path + "not_connected.p", "wb") as fp:
        pickle.dump(not_connected_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


def negative_sampling(path, edge_arr, info, seed=42, num=1):
    print("Negative sampling...")
    with open(path + "not_connected.p", "rb") as fp:
        not_connected_dict = pickle.load(fp)

    negative_edge_arr = list()
    for index in tqdm(range(edge_arr.shape[0])):
        node = edge_arr[index, 0]
        try:
            candidates = not_connected_dict[node]
        except:
            continue
        if len(candidates) != 0:
            if num > len(candidates):
                sample_nodes = candidates 
            else:
                random.seed(seed)
                sample_nodes = random.sample(candidates, k=num)
            not_connected_dict[node] = list(set(candidates) - set(sample_nodes))
            for sample in sample_nodes:
                negative_edge_arr.append([node, sample])

    diff = edge_arr.shape[0] - len(negative_edge_arr)
    while diff > 0:
        node = random.choice(list(not_connected_dict.keys()))
        if len(not_connected_dict[node]) != 0:
            sample_node = random.choice(not_connected_dict[node])
            negative_edge_arr.append([node, sample_node])
            not_connected_dict[node].remove(sample_node) 
            diff -= 1
        else:
            continue

    with open(path + "not_connected.p", "wb") as fp:
        pickle.dump(not_connected_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    negative_edge_arr = np.array(negative_edge_arr)
    y_arr = np.concatenate((np.ones(edge_arr.shape[0]), np.zeros(negative_edge_arr.shape[0])))
    edge_arr = np.concatenate((np.concatenate((edge_arr, negative_edge_arr), axis=0), np.expand_dims(y_arr, axis=1)), axis=1).astype(np.int64)
    np.random.seed(seed)
    np.random.shuffle(edge_arr)
    with open(path + f"{info}_data.npy", "wb") as f:
        np.save(f, edge_arr)
    print("Done.")


class LinkPred(BaseModel):
    """
    Implement algorithms for link prediction task.
    Args:
        data: Training graph data.
        num_nodes: Number of nodes in the graph.
        train_data: Link training data.
        val_data: Link validation data.
        test_data: Link test data.
        oper: Obtain edge features. 1 - average, 2 - hadamard, 3 - L1 norm, 4 - L2 norm.
        lr: Learning rate of optimizer. Default: 0.001.
        node2vec_epoch: The number of epochs of training node2vec model. Default: 50.
        batch_size: The size of a single batch. Default: 128.
        model_batch: Save model. Default: "".
        Refer to https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html#Node2Vec
        for other parameters.
    """
    def __init__(self, data, num_nodes, train_data, val_data, test_data, oper, lr=0.001, node2vec_epoch=50, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128, model_path=""):
        self.use_node2vec = True
        self.model_path = model_path
        self.data = data
        self.node2vec_epoch = node2vec_epoch
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.sparse = sparse
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.oper = oper
        self.num_nodes=num_nodes
        self.embedding_dim=embedding_dim
        self.preprocessing(num_nodes, lr, embedding_dim, walk_length, context_size, walks_per_node, 
                num_negative_samples, p, q, sparse, batch_size)

    def _edge_features(self, features, node_arr, oper=1):
        features1 = features[node_arr[:, 0], :]
        features2 = features[node_arr[:, 1], :]
        if oper == 1:
            return (features1 +  features2) / 2
        elif oper == 2:
            return features1 * features2
        elif oper == 3:
            return np.abs(features1 - features2)
        else:
            return (features1 - features2) ** 2
    
    def train(self):
        self.model.eval()
        self.z = self.model().detach().cpu().numpy()
        train_node_arr = self.train_data[:, [0, 1]]
        train_y = self.train_data[:, 2]
        train_X = self._edge_features(self.z, train_node_arr, self.oper)
        self.clf = LogisticRegression()
        self.clf.fit(train_X, train_y)
    
    def tuning(self, info=""):
        def objective(config):
            self.preprocessing(self.num_nodes, config["lr"], config["embedding_dim"], config["walk_length"], 
                    self.context_size, self.walks_per_node, self.num_negative_samples, config["p"], config["q"], 
                    self.sparse, config["batch_size"])
            for j in tqdm(range(10)):
                self.node2vec_train()
            self.train()
            val_node_arr = self.val_data[:, [0, 1]]
            y_true = self.val_data[:, 2]
            val_X = self._edge_features(self.z, val_node_arr, self.oper)
            y_pred = self.clf.predict_proba(val_X)[:, 1]
            auc = roc_auc_score(y_true, y_pred)
            tune.report(mean_auc=auc)

        search_space = {
            "lr": tune.grid_search([0.001, 0.01, 0.1]),
            "embedding_dim": tune.choice([64, 128]),
            "walk_length": tune.choice([10, 20, 30]),
            "p": tune.choice([0.25, 0.5, 1, 2, 4]),
            "q": tune.choice([0.25, 0.5, 1, 2, 4]),
            "batch_size": tune.choice([64, 128]),
        }

        tuner = tune.Tuner( 
                objective, 
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    num_samples=10,
                    scheduler=ASHAScheduler(metric="mean_auc", mode="max")),
                    run_config=air.RunConfig(local_dir="../results", name=f"{info}_node2vec_tuning")
                )
        results = tuner.fit()

    @torch.no_grad()
    def test(self):
        test_node_arr = self.test_data[:, [0, 1]]
        y_true = self.test_data[:, 2]
        test_X = self._edge_features(self.z, test_node_arr, self.oper)
        y_pred = self.clf.predict_proba(test_X)[:, 1]
        auc = roc_auc_score(y_true, y_pred)
        return auc

    @cal_time
    def main(self):
        if self.use_node2vec == True:
            if os.path.exists(self.model_path):
                self.model = torch.load(self.model_path)
            else:
                print("Training node2vec model...")
                for epoch in tqdm(range(self.node2vec_epoch)):
                    self.node2vec_train()
        print("Training model for the task...")
        self.train()
        print("Evaluating model...")
        auc = self.test()
        print(f"Test Auc: {auc:.4f}")


class LinkPredGCN(LinkPred):
    """
    Implement algorithms for link prediction task.
    Args:
        data: Training graph data.
        num_nodes: Number of nodes in the graph.
        train_data: Link training data.
        val_data: Link validation data.
        test_data: Link test data.
        proposed: Use our method. Default: False.
        gnn_epoch: The number of epochs of training GCN model. Default: 10.
        gnn_lr: Learning rate of GCN. Default: 0.01.
        oper: Obtain edge features. 1 - average, 2 - hadamard, 3 - L1 norm, 4 - L2 norm.
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
            super(LinkPredGCN.GCN, self).__init__()
            self.embedding = torch.nn.Embedding(num_embeddings=node_number, embedding_dim=embedding_dim)
            self.gcn = GCNConv(in_channels=embedding_dim, out_channels=embedding_dim)
            if proposed == True:
                self.embedding.from_pretrained(meta_info, freeze=False)
        
        def forward(self, nodes, edges):
            nodes_embedding = self.embedding(nodes)
            nodes_embedding = self.gcn(nodes_embedding, edges)
            return nodes_embedding

    def __init__(self, data, num_nodes, train_data, val_data, test_data, oper, proposed=False, gnn_epoch=10, gnn_lr=0.01, visual=False, lr=0.001, node2vec_epoch=50, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128, model_path=""):
        super().__init__(data, num_nodes, train_data, val_data, test_data, oper, lr, node2vec_epoch, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, p, q, sparse, batch_size, model_path)
        self.proposed = proposed
        self.gnn_epoch = gnn_epoch
        self.gnn_lr = gnn_lr
        self.visual = visual
        if not proposed:
            self.use_node2vec = False
    
    def train(self):
        self.model.eval()
        self.z = self.model().detach().cpu()
        train_y = self.train_data[:, 2]
        train_node_arr = self.train_data[:, [0, 1]]
        train_y = torch.tensor(train_y, dtype=torch.float32)
        if self.num_nodes is None:
            num_nodes = self.model().shape[0]
        else:
            num_nodes = self.num_nodes
        self.node_arr = torch.arange(num_nodes, dtype=int)
        self.clf = LinkPredGCN.GCN(node_number=num_nodes, embedding_dim=self.embedding_dim, proposed=self.proposed, meta_info=self.z)
        self.clf.train()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.clf.parameters(), lr=self.gnn_lr)
        for i in range(self.gnn_epoch):
            y_pred = self.clf(self.node_arr, self.data.edge_index)
            y_pred = self._edge_features(y_pred, train_node_arr, oper=2)
            y_pred = y_pred.mean(axis=1)
            y_pred = torch.sigmoid(y_pred)
            loss = loss_fn(y_pred, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def gcn_tuning(self, lr, info):
        self.gnn_lr = lr
        if self.use_node2vec == True:
            if os.path.exists(self.model_path):
                self.model = torch.load(self.model_path)
            else:
                print("Training node2vec model...")
                for epoch in tqdm(range(self.node2vec_epoch)):
                    self.node2vec_train()
        print("Training model for the task...")
        self.train()

        @torch.no_grad()
        def evaluate():
            print("Evaluating model...")
            val_node_arr = self.val_data[:, [0, 1]]
            val_X = self._edge_features(self.z, val_node_arr, 2)
            y_true = self.val_data[:, 2]  
            y_true = torch.tensor(y_true)
            y_pred = self.clf(self.node_arr, self.data.edge_index)
            y_pred = self._edge_features(y_pred, val_node_arr, 2)
            y_pred = y_pred.mean(axis=1)
            y_pred = torch.sigmoid(y_pred)
            auc = roc_auc_score(y_true, y_pred)
            print(f"{info}\nAUC: {auc}")

        evaluate()

    @torch.no_grad()
    def test(self):
        test_node_arr = self.test_data[:, [0, 1]]
        test_X = self._edge_features(self.z, test_node_arr, 2)
        y_true = self.test_data[:, 2]  
        y_true = torch.tensor(y_true)
        y_pred = self.clf(self.node_arr, self.data.edge_index)
        y_pred = self._edge_features(y_pred, test_node_arr, 2)
        y_pred = y_pred.mean(axis=1)
        y_pred = torch.sigmoid(y_pred)
        auc = roc_auc_score(y_true, y_pred)
        return auc
