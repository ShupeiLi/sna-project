## Slide 1: Title
Hello, everyone! We are group 43. The paper we are going to present today is node2vec: scalable feature learning for networks, written by Grover and Leskovec.

## Slide 2: Contents
This is a brief content of our presentation. Firstly, we will introduce some background knowledge about graph embeddings, followed by a review of related work. Then, we will explain the node2vec algorithm in detail and present experimental results in the original paper. After that, we will introduce the idea of our course project. This presentation is concluded with the future work we plan to do in the project.

## Slide 3: Introduction to Graph Embeddings
As we've seen in this course, complex relationships can be modelled by graphs. Graph-structured data appears in many modern applications like molecular structure modelling, social network analysis, recommender system, etc. However, if we want to leverage the information contained in graphs, we need to find a way to represent graph-structured data efficiently. Traditional statistical and machine learning methods are designed for extracting features from structured data, such as PCA, UMAP, t-SNE. Although these methods have achieved satisfactory performance on structured-data, they are hard to be generalized to graph-structured data, because they higly depend on properties of Eucildean space. This challenge led to the development of techniques specially for graph representation. Figure on the slide illustrates the general idea of graph embedding learning method. Firstly, we obtain embeddings by mapping graphs onto a low dimension space. And then, we can use these features in some downstream tasks like node classification and link prediction. From this perspective, node2vec algorithm that we address in presentation is a feature learning framework.

## Slide 4: Related Work
This figure illustrates the taxonomy of graph embedding techniques, according to a textbook written by Murphy. There are two types of graph embedding techniques. They are shallow embedding method and deep embedding method. Shallow embedding methods use shallow encoding functions to map the original graph structure onto a Euclidean space and obtain the embedding matrix. If data is labelled, we can apply supervised algorithms to extract embeddings, for example, label propagation. However, labels are not avaliable or only partly avaliable in most cases, where we need to use unspervised or semi-supervised learning. These methods can be divided into distance-based method and outer product method further. Some representative algorithms are given in the figure. It is worth noting that node2vec algorithm we emphasize in presentation belongs to outer product-based method using skip-gram architecture, which is inspired by skip-gram model in natral language processing field. Deep embedding method usually refers to algorithms that learn graph features via graph neural networks, or GNN. GNN is a special class of artificial neural networks designed for graph-structured data. In our project experiments, we mainly use a method called graph convolutional networks, which is a popular GNN architecture.

## Slide 15: Experiment 1: Multi-label Classification (1/2)
Authors of the original paper conducted two experiments to verify the effectiveness of the node2vec algorithm. The first experiment is multi-label classification. Given a finite label set and a fraction of nodes with labels, the task is predicting the labels for the remaining nodes. They use three data sets: BlogCatalog, PPI, and Wikipedia. The table shows some statistics of data sets. Macro-F1 score is selected as metrics.

## Slide 16: Experiment 1: Multi-label Classification (2/2)
Spectral clustering, DeepWalk, and LINE are selected as benchmarks. From the table, we can see that node2vec outperforms other algorithms on all datasets. Besides, its relative performance gain is significant.

## Slide 17: Experiment 2: Link Prediction (1/2)
The second task is link prediction. Given a network with a fraction of edges removed, the task is predicting these missing edges. They also use three datasets, Facebook, PPI, and arXiv, in this experiment. Statistics of data sets are shown in the table. AUC score is used as metrics.

## Slide 18: Experiment 2: Link Prediction (2/2)
When the original paper was published, there were no feature learning algorithms that had been used for link prediction. Therefore, authors set four heuristic scores as benchmarks. The first four lines of table shows the performance of heuristic scores. The other three benchmarks are the same as in the experiment 1. Authors tested average, hadamard, weighted-L1, and weighted-L2 operations on these three benchmarks as well as node2vec. Due to the limited space, table on the slide only shows the result of hadamard operation. We can see that the learned feature representations outperform heuristic scores. And node2vec achieves the best AUC score on all datasets.

## Slide 19: Summary of node2vec
A brief summary of node2vec. It is an algorithm designed for extracting embedidngs from graphs. Results of experiments support the effectiveness of the algorithm. Its neighborhood search strategy is flexible and controllable. We can easily change the behavior of search by adjusting hyperparameter p and q.

## Slide 20: Our Contributions
Next, I will talk about our contributions in the project. Firstly, the intermediate state of node embeddings during node2vec training is a black box. We try to visualize the node embeddings with t-SNE technique to provide some insights into node2vec. Secondly, we find that researchers in graph embedding learning field have focused more on deep embedding methods these years. However, randomly initialized inputs in GNN affect the robustness of model performance and extend the training time. To address the problem, we proposed a novel method that combines node2vec and GNN. Thirdly, to further test the effectiveness of algorithms, we evaluate node2vec, GNN, and our proposed method on five real world datasets and select different metrics.

## Slide 21: Visualization
We apply node2vec algorithm on an open source dataset AIFB. Results of visualization are showed on the slide. Each node in AIFB belongs to a class. The total number of classes is four. We set the number of epochs to 100. It is easy to see that nodes embeddings are changed from chaos into order during the training of node2vec.

## Slide 22: Proposed Method
As mentioned before, random initialization in GNN may not be a good strategy. Inspired by the concept of meta learning, we use the pretrained embeddings from node2vec as the meta information for GNN. Figure on the right illustrates the architecture of our proposed model. We expect our method will accelerate the training stage of GNN and improve the model performance. We choose graph convolutional network or GCN in our experiments. The iteration formula of GCN is given on the slide. Preliminary results we have obtained so far support the effectiveness of our proposed method.

## Slide 23: Future work
Our future work will mainly focus on two tasks. Task one, apply node2vec, GCN, and our proposed model to node classification and link prediction. Task two, try different strategies of hyperparameter tuning. Due to the time limit, we omit technical details of GNN. If you are interested in GNN, we can discuss the principle of GNN together. Thank you. This's all of our presentation. Any questions?
