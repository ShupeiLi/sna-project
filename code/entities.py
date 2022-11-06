# -*- coding: utf-8 -*-

from node2vec import BaseModel
import torch
import numpy as np
from torch_geometric.datasets import Entities
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch.nn as nn
from torch_geometric.nn import GCNConv


aifb_data = Entities(root="../data/Entities/AIFB/", name="AIFB")
aifb_data = aifb_data[0]
#am_data = Entities(root="../data/Entities/AM/", name="AM")
#am_data = am_data[0]
#mutag_data = Entities(root="../data/Entities/MUTAG/", name="MUTAG")
#mutag_data = mutag_data[0]
#bgs_data = Entities(root="../data/Entities/BGS/", name="BGS")
#bgs_data = bgs_data[0]

class GCN_Entity(nn.Module):
    def __init__(self, nodes_number,embedding_dim, metainfor, use_metainfor):
        super(GCN_Entity,self).__init__()
        self.embedding=torch.nn.Embedding(num_embeddings=nodes_number,embedding_dim=embedding_dim)
        self.gcn=GCNConv(in_channels=embedding_dim,out_channels=embedding_dim)
        if use_metainfor==True:
            self.embedding.from_pretrained(metainfor,freeze=False)
    
    def forward(self, nodes, edges):
        nodes_embedding=self.embedding(nodes)
        nodes_embedding=self.gcn(nodes_embedding,edges)
        nodes_embedding=torch.relu(nodes_embedding)
        return nodes_embedding

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
    def __init__(self, data, gnn=True, use_metainfor=True , gnn_epoch=10, multi=True, lr=0.001, epochs=50, embedding_dim=128, walk_length=20, context_size=10,
            walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True, batch_size=128):
        self.data = data
        self.data.nodes=torch.tensor(range(data.num_nodes))
        self.multi = multi
        self.epochs = epochs
        self.gnn_epoch=gnn_epoch
        self.gnn=gnn 
        self.use_metainfor=use_metainfor
        self.embedding_dim=embedding_dim
        self.preprocessing(data.num_nodes, lr, embedding_dim, walk_length, context_size, walks_per_node, 
                num_negative_samples, p, q, sparse, batch_size)


    def down_stream_train(self):
        self.model.eval()
        self.z = self.model()
        
        
        if self.gnn == False:
            self.clf = LogisticRegression()
            self.clf.fit(self.z[self.data.train_idx].detach().cpu().numpy(), self.data.train_y.detach().cpu().numpy())
        
        if self.gnn==True:
            if self.use_metainfor==False:
                self.clf=GCN_Entity(nodes_number=self.data.num_nodes, embedding_dim=self.embedding_dim,use_metainfor=self.use_metainfor, metainfor=self.z)
                self.clf.train()
                loss_fn=torch.nn.CrossEntropyLoss()
                optimizer=torch.optim.Adam(params=self.clf.parameters(),lr=0.01)
                for i in range(self.gnn_epoch):
                    y_pred = self.clf(self.data.nodes, self.data.edge_index)
                    loss=loss_fn(y_pred[self.data.train_idx],self.data.train_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            if self.use_metainfor==True:
                self.clf=GCN_Entity(nodes_number=self.data.num_nodes, embedding_dim=self.embedding_dim,use_metainfor=self.use_metainfor, metainfor=self.z)
                self.clf.train()
                loss_fn=torch.nn.CrossEntropyLoss()
                optimizer=torch.optim.Adam(params=self.clf.parameters(),lr=0.01)
                for i in range(self.gnn_epoch):
                    y_pred = self.clf(self.data.nodes, self.data.edge_index)
                    loss=loss_fn(y_pred[self.data.train_idx],self.data.train_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
   
    @torch.no_grad()   
    def test(self):
        self.clf.eval()
            
        if self.gnn==False:
            y_pred = self.clf.predict(self.z[self.data.test_idx].detach().cpu().numpy())
            y_true = self.data.test_y.detach().cpu().numpy()
            acc = accuracy_score(y_true, y_pred)
            if self.multi:
                recall = recall_score(y_true, y_pred, labels=np.unique(y_true), average="weighted")
                f1 = f1_score(y_true, y_pred, labels=np.unique(y_true), average="weighted")
            else:
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            return acc, recall, f1
        
        if self.gnn==True:
            y_pred = self.clf(self.data.nodes, self.data.edge_index).detach().cpu().numpy()
            y_pred = y_pred[self.data.test_idx].argmax(1)
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
