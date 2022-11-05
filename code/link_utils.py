# -*- coding: utf-8 -*-

import pickle
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from node2vec import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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
        test_data: Link test data.
        oper: Obtain edge features. 1 - average, 2 - hadamard, 3 - L1 norm, 4 - L2 norm.
        lr: Learning rate of optimizer. Default: 0.001.
        epochs: The number of epochs of training. Default: 50.
        batch_size: The size of a single batch. Default: 128.
        Refer to https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html#Node2Vec
        for other parameters.
    """
    def __init__(self, data, num_nodes, train_data, test_data, oper, lr=0.001, epochs=50, embedding_dim=128, walk_length=20, context_size=10,
            walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128):
        self.data = data
        self.epochs = epochs
        self.train_data = train_data
        self.test_data = test_data
        self.oper = oper
        self.preprocessing(num_nodes, lr, embedding_dim, walk_length, context_size, walks_per_node, 
                num_negative_samples, p, q, sparse, batch_size)

    def _edge_features(self, features, node_arr, oper=1):
        features1 = features[node_arr[:, 0], :]
        features2 = features[node_arr[:, 1], :]
        if oper == 1:
            return (features1 +  features2) / 2
        elif oper == 2:
            return feature1 * feature2
        elif oper == 3:
            return np.abs(feature1 - feature2)
        else:
            return (feature1 - feature2) ** 2

    @torch.no_grad()
    def test(self):
        self.model.eval()
        z = self.model().detach().cpu().numpy()
        
        train_node_arr = self.train_data[:, [0, 1]]
        train_y = self.train_data[:, 2]
        train_X = self._edge_features(z, train_node_arr, self.oper)
        
        test_node_arr = self.test_data[:, [0, 1]]
        y_true = self.test_data[:, 2]
        test_X = self._edge_features(z, test_node_arr, self.oper)

        clf = LogisticRegression()
        clf.fit(train_X, train_y)
        y_pred = clf.predict_proba(test_X)[:, 1]
        auc = roc_auc_score(y_true, y_pred)
        return auc

    @cal_time
    def main(self):
        if not hasattr(self, "epochs"):
            raise AttributeError(f"Class attribute 'epochs' is not defined.")
        for epoch in range(1, self.epochs + 1):
            loss = self.train()
            auc = self.test()
            print(f"Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test Auc: {auc:.4f}.")
