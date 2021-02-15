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
 embedder = KGEmbedder(graphs,'examples/',clear_dir=True)
 embeddings = embedder.embed()

```
