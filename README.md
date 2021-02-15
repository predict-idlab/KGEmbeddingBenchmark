# KGEmbeddingBenchmark
Framework for embedding whole Knowledge Graphs and compare various techniques.

## Knowledge Graph Comparison

This frameworks allows to generate and compare whole Knowledge Graph embeddings based on six graph embedding techniques:

- [Graph2Vec](https://github.com/benedekrozemberczki/graph2vec)
- [GE-FSG](https://epubs.siam.org/doi/abs/10.1137/1.9781611975321.35)
- [Deep Graph Kernels](https://dl.acm.org/doi/abs/10.1145/2783258.2783417)
- [weisfeiler lehman kernel](https://github.com/benedekrozemberczki/graph2vec)
- [SimGNN](https://github.com/yunshengb/SimGNN)
- [Graph Matching Networks](https://github.com/deepmind/deepmind-research/tree/master/graph_matching_networks)


## Usage:

```python
 graphs = ...
 # Graph2Vec
 embedder = KGEmbedder(graphs,'temp_dir/',clear_dir=True)
 # GE-FSG
 embedder = GEFSGEmbedder(graphs,'temp_dir/',min_sup=2,clear_dir=True)
 # DGK
 embedder = DeepGraphEmbedder(graphs)
 # WL
 embedder = WLEmbedder(graphs,'temp_dir/')
 # SimGNN
 # requires train and test set of graphs with their GED
 embedder = SimGNNEmbedder(graphs_train,graphs_test,output_dir_train,output_dir_test,relable=True,clear_dir=True)
 # GMN
 embedder = GMNEmbedder(graphs)
 
 embeddings = embedder.embed()

```
