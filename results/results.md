# AIFB
- node2vec: 10 epochs, 0.1 lr, 30 walk length, 0.25 p, 0.25 q, 128 embedding dims, 128 batch size.
- GCN: 50 epochs, 2 layers 128-64-32, 0.01 lr, relu, dropout 0.5.

||node2vec|GCN|Proposed|
|---|---|---|---|
|ACC|0.9444|0.9444|0.9722|
|Recall|0.8750|0.9375|0.9583|
|Maro-F1|0.9116|0.9508|0.9416|

# Mutag
- node2vec: 20 epoch, 0.01 lr, 30 walk length, 2 p, 4 q, 128 embedding dims, 128 batch size.
- GCN: 50 epochs, 3 layers 64-32-16, 0.01 lr, relu, dropout 0.5-0.1.

||node2vec|GCN|Proposed|
|---|---|---|---|
|ACC|0.7941|0.7353|0.6912|
|Recall|0.5652|0.6193|0.7242|
|Maro-F1|0.6500|0.6151|0.6857|

# Pubmed
- node2vec: 10 epoch, 0.01 lr, 20 walk length, 4 p, 4 q, 64 embedding dims, 64 batch size.
- GCN: 50 epoch, 3 layers 32-16-8, 0.05 (GCN) / 0.1 (Our) lr, relu, dropout 0.1-0.1.

||node2vec|GCN|Proposed|
|---|---|---|---|
|ACC|0.6650|0.6770|0.6920|
|Recall|0.6434|0.6443|0.6508|
|Maro-F1|0.6438|0.6482|0.6616|

# Rellink
||node2vec|GCN|Proposed|
|---|---|---|---|
|ACC||||

# Wordnet18rr
||node2vec|GCN|Proposed|
|---|---|---|---|
|ACC||||
