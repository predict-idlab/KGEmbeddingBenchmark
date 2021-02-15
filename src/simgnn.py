from graphgen import GraphGen
import random
import rdflib
import glob
import os
import networkx as nx
import json
from tqdm import tqdm
from utils.simgnn.simgnn import SimGNNTrainer
from kgcompare import KGCompare
import numpy as np


class Args():
    def __init__(self,output_dir_train,output_dir_test):
        self.training_graphs = output_dir_train
        self.testing_graphs= output_dir_test
        self.epochs = 10
        self.filters_1 =  128
        self.filters_2= 64
        self.filters_3 =  32
        self.tensor_neurons = 16
        self.bottle_neck_neurons = 16
        self.batch_size =32
        self.bins = 16
        self.dropout = 0.5
        self.learning_rate =  0.001
        self.weight_decay = 5 * 10 ** -4
        self.histogram = False

class SimGNNEmbedder():
    def __init__(self,graphs_train,graphs_test,output_dir_train,output_dir_test,relable=True,clear_dir=False,args=None):
        self.graphs_train=graphs_train
        self.graphs_test = graphs_test
        self.embeddings=[]


        self.output_dir_train = output_dir_train
        self.output_dir_test = output_dir_test
        self.label_indexes = {}
        if clear_dir:
            self.clear_intermediate_files(self.output_dir_train)
            self.clear_intermediate_files(self.output_dir_test)
        if relable:
            print('converting train graphs')
            for graph_index,graph_pair in enumerate(graphs_train):
                self.write_graphs_to_file(self.output_dir_train,graph_pair[0],graph_pair[1],graph_index)
            print('converting test graphs')
            for graph_index,graph_pair in enumerate(graphs_test):
                self.write_graphs_to_file(self.output_dir_test,graph_pair[0],graph_pair[1],graph_index)
        if args == None:
            args = Args(output_dir_train,output_dir_test)
        self.model = SimGNNTrainer(args)
    def clear_intermediate_files(self,output_dir):
        files = glob.glob(output_dir+'*')
        for f in files:
            os.remove(f)

    def approximate_ged(self,graph1,graph2):
        '''
        Approximates the GED distance by counting the sum of the triples that should be removed from graph1 and added to convert it into graph2.
        '''
        g1_set = set(graph1.triples((None, None, None)))
        g2_set = set(graph2.triples((None, None, None)))
        g1_deletes = list(g1_set - g2_set)
        g1_adds = list(g2_set - g1_set)
        return len(g1_deletes) + len(g1_adds)

    def embed(self):
        print('start training')
        self.model.fit()
        print('start predictions')
        self.predictions = self.model.score()
        return self.predictions

    def write_graphs_to_file(self,output_dir,graph1,graph2,graph_index):
        json_graph = self.to_json(graph1, graph2, label_indexes=self.label_indexes)
        with open(output_dir + str(graph_index) + '.json', 'w') as f:
            json.dump(json_graph, f)

    def to_json(self,graph1,graph2,label_indexes = None):
        if label_indexes == None:
            label_indexes=self.label_indexes
        gen = GraphGen()
        gen.graph = graph1
        gen2 = GraphGen()
        gen2.graph = graph2
        g1 = gen.convert_to_nx(relable=True, num_values=True,label_indexes=label_indexes)
        g2 = gen2.convert_to_nx(relable=True, num_values=True,label_indexes=label_indexes)

        json = {}
        json['graph_1'] = [[int(e[0]), int(e[1])] for e in list(g1.edges)]
        json['graph_2'] = [[int(e[0]), int(e[1])] for e in list(g2.edges)]
        labels1 = gen.nx_labels
        json['labels_1'] = [labels1[i] for i in range(len(list(g1.nodes)))]
        labels2 = gen2.nx_labels
        json['labels_2'] = [labels2[i] for i in range(len(list(g2.nodes)))]

        #compute ged
        ged = self.approximate_ged(graph1,graph2)
        json['ged'] = ged
        return json

class SimGNNKGCompare(KGCompare):
    def __init__(self,embedder):
        self.embedder=embedder

    def get_dist_matrix(self):
        """computes distances matrix based on the embedding in the KGEmbedder"""
        distances = np.zeros((len(self.embedder.predictions),len(self.embedder.predictions)))
        distances[0] = self.embedder.predictions
        return distances

def read_graphs(graphs_dir):
    graph1 = rdflib.Graph()
    graph1.parse(graphs_dir[0], format="ttl")
    graph2 = rdflib.Graph()
    graph2.parse(graphs_dir[1], format="ttl")
    return (graph1,graph2)
def main():
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt

    output_dir_train = 'examples_simgnn/train/'
    output_dir_test = 'examples_simgnn/test/'
    num_train = 20
    num_test=20

    graphs_folder = '../KGComparisonBenchmark/data/mutag/rdf_graphs/'
    graph_files = [graphs_folder+str(item)+'.ttl'
                                   for item in range(len(glob.glob(graphs_folder + '/*.ttl')))
                                   if os.path.isfile(graphs_folder+str(item)+'.ttl')]
    graphs=[rdflib.Graph().parse(graph_file, format="ttl") for graph_file in graph_files]
    graphs_train=[]
    for i in range(num_train):
        graphs_train.append(read_graphs(random.sample(graph_files, 2)))

    graphs_test=[]
    for i in range(num_test):
        graphs_test.append(read_graphs(random.sample(graph_files, 2)))

    simgnnEmbedder = SimGNNEmbedder(graphs_train,graphs_test,output_dir_train,output_dir_test,relable=True,clear_dir=True)
    result = simgnnEmbedder.embed()
    compare = SimGNNKGCompare(simgnnEmbedder)
    m=compare.get_dist_matrix()

    geds=[]
    for i in range(num_test):
        with open(output_dir_test+str(i)+'.json') as json_file:
            data = json.load(json_file)
            geds.append(data['ged'])

    scaler = MinMaxScaler()
    norm = scaler.fit_transform(np.array(m[0]).reshape(-1,1))
    norm2 = scaler.fit_transform(np.array(geds).reshape(-1,1))
    plt.plot(norm,label='predicted')
    plt.plot(norm2,label='real')
    plt.legend()
    plt.show()

def gradual_changing_graphs():
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt

    output_dir_train = 'examples_simgnn/train/'
    output_dir_test = 'examples_simgnn/test/'
    num_train = 100

    graphs_folder = '../KGComparisonBenchmark/data/mutag/rdf_graphs/'
    graph_files = [graphs_folder + str(item) + '.ttl'
                   for item in range(len(glob.glob(graphs_folder + '/*.ttl')))
                   if os.path.isfile(graphs_folder + str(item) + '.ttl')]
    selected_graph_files = random.sample(graph_files, 2)
    #remove the selected graphs from the possible graphs
    graph_files.remove(selected_graph_files[0])
    graph_files.remove(selected_graph_files[1])

    graph1 = rdflib.Graph()
    graph1.parse(selected_graph_files[0], format="ttl")
    graph2 = rdflib.Graph()
    graph2.parse(selected_graph_files[1], format="ttl")
    gen = GraphGen()
    gen.graph = graph1
    graphs = gen.convert_into(graph2, ignore_del=False)


    num_graphs = 10
    selected_graphs = [graphs[i] for i in range(0, len(graphs), int(len(graphs) / num_graphs))]
    graphs_train=[]
    for i in range(int(num_train)):
        graphs_train.append(read_graphs(random.sample(graph_files, 2)))

    graphs_test=[]
    for i in range(num_graphs):
        graphs_test.append((selected_graphs[0],selected_graphs[i]))

    simgnnEmbedder = SimGNNEmbedder(graphs_train,graphs_test,output_dir_train,output_dir_test,relable=True,clear_dir=True)
    result = simgnnEmbedder.embed()
    compare = SimGNNKGCompare(simgnnEmbedder)
    m=compare.get_dist_matrix()

    geds=[]
    for i in range(num_graphs):
        with open(output_dir_test+str(i)+'.json') as json_file:
            data = json.load(json_file)
            geds.append(data['ged'])

    scaler = MinMaxScaler()
    norm = scaler.fit_transform(np.array(m[0]).reshape(-1,1))
    norm2 = scaler.fit_transform(np.array(geds).reshape(-1,1))
    plt.plot(norm,label='predicted')
    plt.plot(norm2,label='real')
    plt.legend()
    plt.show()
