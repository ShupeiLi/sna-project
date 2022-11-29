# -*- coding: utf-8 -*-

from link_utils import *
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import RelLinkPredDataset


path = "../data/RelLink/"
rellink_data = RelLinkPredDataset(root=path, name="FB15k-237")
rellink_data = rellink_data[0]
edge_arr = np.transpose(rellink_data.edge_index.detach().cpu().numpy())
node_arr = np.unique(edge_arr.flatten())


def create_datasets(rellink_data, edge_arr, node_arr):
    train_edge_arr = np.transpose(rellink_data.train_edge_index.detach().cpu().numpy())
    valid_edge_arr = np.transpose(rellink_data.valid_edge_index.detach().cpu().numpy())
    test_edge_arr = np.transpose(rellink_data.test_edge_index.detach().cpu().numpy())

    not_connected_store(path, edge_arr, node_arr)
    negative_sampling(path, train_edge_arr, "train", seed=42, num=1)
    negative_sampling(path, valid_edge_arr, "valid", seed=42, num=1)
    negative_sampling(path, test_edge_arr, "test", seed=42, num=1)


#create_datasets(rellink_data, edge_arr, node_arr)
train_data = np.load(path + "train_data.npy")
val_data = np.load(path + "valid_data.npy")
test_data = np.load(path + "test_data.npy")
train_graph = Data(x=torch.tensor(np.expand_dims(node_arr, axis=1)), edge_index=rellink_data.train_edge_index)


if __name__ == "__main__":
#   rellink_model = LinkPred(train_graph, None, train_data, val_data, test_data, 1, node2vec_epoch=5)
    for oper in range(1, 5):
        rellink_model = LinkPred(train_graph, None, train_data, val_data, test_data, oper, node2vec_epoch=5)
        rellink_model.tuning(f"rellink_{oper}")
#   rellink_model.main()
#   rellink_gcn_model = LinkPredGCN(train_graph, None, train_data, val_data, test_data, 1, proposed=False, gnn_epoch=10)
#   rellink_gcn_model.main()
