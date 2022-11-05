# -*- coding: utf-8 -*-

from node2vec import BaseModel
import torch
import numpy as np
from torch_geometric.datasets import Entities
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score


aifb_data = Entities(root="../data/Entities/AIFB/", name="AIFB")
aifb_data = aifb_data[0]
#am_data = Entities(root="../data/Entities/AM/", name="AM")
#am_data = am_data[0]
#mutag_data = Entities(root="../data/Entities/MUTAG/", name="MUTAG")
#mutag_data = mutag_data[0]
#bgs_data = Entities(root="../data/Entities/BGS/", name="BGS")
#bgs_data = bgs_data[0]


class Entities(BaseModel):
    """
    Implement algorithms with entities data.
    Args:
        data: Graph data.
        multi: Multiclass classification or binary class classification. Default: True.
        lr: Learning rate of optimizer. Default: 0.001.
        epochs: The number of epochs of training. Default: 50.
        batch_size: The size of a single batch. Default: 128.
        Refer to https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html#Node2Vec
        for other parameters.
    """
    def __init__(self, data, multi=True, lr=0.001, epochs=50, embedding_dim=128, walk_length=20, context_size=10,
            walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128):
        self.data = data
        self.multi = multi
        self.epochs = epochs
        self.preprocessing(data.num_nodes, lr, embedding_dim, walk_length, context_size, walks_per_node, 
                num_negative_samples, p, q, sparse, batch_size)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        z = self.model()
        clf = LogisticRegression()
        clf.fit(z[self.data.train_idx].detach().cpu().numpy(), self.data.train_y.detach().cpu().numpy())
        y_pred = clf.predict(z[self.data.test_idx].detach().cpu().numpy())
        y_true = self.data.test_y.detach().cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        if self.multi:
            recall = recall_score(y_true, y_pred, labels=np.unique(y_true), average="weighted")
            f1 = f1_score(y_true, y_pred, labels=np.unique(y_true), average="weighted")
        else:
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        return acc, recall, f1


if __name__ == "__main__":
    aifb_model = Entities(aifb_data, epochs=5)
    aifb_model.main()
#   am_model = Entities(am_data, epochs=5)
#   am_model.main()
#   mutag_model = Entities(mutag_data, multi=False, epochs=5)
#   mutag_model.main()
#   bgs_model = Entities(bgs_data, multi=False, epochs=5)
#   bgs_model.main()
