# -*- coding: utf-8 -*-

from link_utils import *
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import WordNet18RR


path = "../data/WordNet18RR/"
wordnet_data =  WordNet18RR(root=path)
wordnet_data = wordnet_data[0]
edge_arr = np.transpose(wordnet_data.edge_index.detach().cpu().numpy())
node_arr = np.unique(edge_arr.flatten())


def create_datasets(wordnet_data, edge_arr, node_arr):
    train_edge_arr = np.transpose(wordnet_data.edge_index[:, wordnet_data.train_mask].detach().cpu().numpy())
    valid_edge_arr = np.transpose(wordnet_data.edge_index[:, wordnet_data.val_mask].detach().cpu().numpy())
    test_edge_arr = np.transpose(wordnet_data.edge_index[:, wordnet_data.test_mask].detach().cpu().numpy())

    not_connected_store(path, edge_arr, node_arr)
    negative_sampling(path, train_edge_arr, "train", seed=42, num=1)
    negative_sampling(path, valid_edge_arr, "valid", seed=42, num=1)
    negative_sampling(path, test_edge_arr, "test", seed=42, num=1)


#create_datasets(wordnet_data, edge_arr, node_arr)
train_data = np.load(path + "train_data.npy")
val_data = np.load(path + "valid_data.npy")
test_data = np.load(path + "test_data.npy")
train_graph = Data(x=torch.tensor(np.expand_dims(node_arr, axis=1)), edge_index=wordnet_data.edge_index[:, wordnet_data.train_mask])


if __name__ == "__main__":
#   wordnet_model = LinkPred(train_graph, node_arr.shape[0], train_data, val_data, test_data, 1, node2vec_epoch=5)
    for oper in range(1, 5):
        wordnet_model = LinkPred(train_graph, node_arr.shape[0], train_data, val_data, test_data, oper, node2vec_epoch=5)
        wordnet_model.tuning(f"wordnet_{oper}")
#   wordnet_model.main()
#   wordnet_gcn_model = LinkPredGCN(train_graph, node_arr.shape[0], train_data, val_data, test_data, 1, proposed=False, gnn_epoch=10)
#   wordnet_gcn_model.main()
