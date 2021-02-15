import converter
import glob
from rdflib import Graph
import time
import deep_kernel
import pickle
class DeepGraphEmbedder:
    def __init__(self, graphs):
        self.graphs = graphs
        self.embeddings = []

    def embed(self):
        K,P,M=deep_kernel.compute_dgk(self.graphs,kernel_type=1)
        self.embeddings=K
        return K


def main():
    graph_locs = '/Users/psbonte/Documents/Projects/Radiance/Data/ML6/Cambridge/context/graphs/'
    names = ['Trumpington', 'Petersfield', 'None', "Queen Edith's", 'Newnham', 'Castle', 'East Chesterton', 'Abbey',
             'Coleridge', 'Cherry Hinton', 'Market', 'West Chesterton', 'Arbury', 'Test']
    graphs = []
    for graph_name in names[:2]:
        g = Graph()
        g.parse(graph_locs + graph_name + ".ttl", format="ttl")
        graphs.append(g)
    start= time.time()
    embedder = DeepGraphEmbedder(graphs)
    embedding = embedder.embed()
    stop = time.time()
    print(stop-start)
    pickle.dump(embedder,open( "examples/embedder_deepgraph.pkl", "wb" ))

if __name__ == '__main__':
    main()