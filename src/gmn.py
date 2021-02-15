import utils.gmn as gmn
import rdflib
import os
import glob
import random
from graphgen import GraphGen
from kgcompare import KGCompare

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
class Embedder():

    def __init__(self,graphs):
        self.graphs=graphs

    def construct_embedding(self,embed_size=4):
        gen = GraphGen()
        labels = {}
        for graph in self.graphs:
            gen.graph = graph
            gen.convert_to_nx(relable=True,num_values=True, label_indexes=labels)
        self.voc_size = len(labels)
        self.labels=labels
        self.model = Sequential()
        self.model.add(Embedding(self.voc_size, embed_size, input_length=1))
        self.embeddings={i:self.model.predict([i]).reshape(-1) for i in range(self.voc_size)}
    def get_embedding(self,input_values):
        return self.embeddings[input_values]



class GMNEmbedder():
    def __init__(self, graphs,iterations=10,min_changes=1,max_changes=10):
        self.graphs= graphs
        test_graph_flattend = []
        for graph in self.graphs:
            test_graph_flattend.append(self.graphs[0])
            test_graph_flattend.append(graph)
        self.graph_test = test_graph_flattend
        self.embedder = Embedder(graphs)
        self.embedder.construct_embedding(embed_size=4)
        self.iterations = iterations
        self.min_changes= min_changes
        self.max_changes = max_changes

    def embed(self):
        self.predictions = -1*gmn.train(self.graphs, self.graph_test, self.embedder,self.iterations,self.min_changes,self.max_changes)
        return self.predictions


class GMNCompare(KGCompare):
    def __init__(self,embedder):
        self.embedder=embedder

    def get_dist_matrix(self):
        """computes distances matrix based on the embedding in the KGEmbedder"""
        distances = np.zeros((len(self.embedder.predictions),len(self.embedder.predictions)))
        distances[0] = self.embedder.predictions
        return distances

def main():

    graphs_folder = '../KGComparisonBenchmark/data/mutag/rdf_graphs/'
    graph_files = [graphs_folder + str(item) + '.ttl'
                   for item in range(len(glob.glob(graphs_folder + '/*.ttl')))
                   if os.path.isfile(graphs_folder + str(item) + '.ttl')]
    graphs = [rdflib.Graph().parse(graph_file, format="ttl") for graph_file in graph_files]

    selected_graph_files = random.sample(graph_files, 2)
    graph1 = rdflib.Graph()
    graph1.parse(selected_graph_files[0], format="ttl")
    graph2 = rdflib.Graph()
    graph2.parse(selected_graph_files[1], format="ttl")
    gen = GraphGen()
    gen.graph = graph1
    graphs = gen.convert_into(graph2, ignore_del=False)


    num_graphs = 10
    #selected_graphs = [graphs[i] for i in range(0, len(graphs), int(len(graphs) / num_graphs))]
    selected_graphs = [graphs[i] for i in range(0, 100)]

    gmnembedder = GMNEmbedder(selected_graphs)
    distances = gmnembedder.embed()

    import matplotlib.pyplot as plt
    plt.plot(distances)
    plt.show()






