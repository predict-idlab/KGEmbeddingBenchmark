from kg2vec import KGEmbedder
from wlembedding import WLEmbedder
from deepgraphembedding import DeepGraphEmbedder
from kg2vec import main as trainEmbed
import pickle
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from kgcompare_stateless import StatelessKGCompare


class KGCompare:
    """Adapter for StatelessKGCompareAPI, such that code referring to KGCompare is still callable as-is."""
    def __init__(self,embedder):
        self.embedder=embedder

    def get_dist_matrix(self,distance_metric=spatial.distance.cosine):
        """computes distances matrix based on the embedding in the KGEmbedder"""
        try:
            return self.dist_matrix
        except AttributeError:
            self.dist_matrix = StatelessKGCompare.get_dist_matrix(self.embedder.embeddings, distance_metric)
            return self.dist_matrix

    def get_generic_similarity(self, graph_id,similarity_measure, distance_metric=spatial.distance.cosine):
        """returns list with most similar graph ids, based on generic similarity measure, either min/max"""
        return StatelessKGCompare.get_generic_similarity(graph_id, self.embedder.embeddings, similarity_measure, distance_metric)

    def get_most_similar(self, graph_id, distance_metric=spatial.distance.cosine):
        """returns list with most similar graph ids"""
        return StatelessKGCompare.get_most_similar(graph_id, self.embedder.embeddings, distance_metric)

    def get_least_similar(self, graph_id, distance_metric=spatial.distance.cosine):
        """returns list with least similar graph ids"""
        return StatelessKGCompare.get_least_similar(graph_id, self.embedder.embeddings, distance_metric)

    def get_graph_overlap(self, graphs):
        try:
            return self.graph_overlap
        except AttributeError:
            self.graph_overlap=StatelessKGCompare.get_graph_overlap(graphs)
            return self.graph_overlap


def visualize_distance_matrix(dist_matrix, title=''):

    plt.matshow(dist_matrix)
    plt.xlabel('Graph IDs 1')
    plt.ylabel('Graphs IDs 2')
    plt.title(title)
    plt.colorbar()
    plt.show()

def visualize_distance_vector(dist_vector,title=''):

    plt.title(title)
    plt.xlabel('elements')
    plt.ylabel('distance')
    plt.plot(dist_vector)
    plt.show()

def po_match(G1,G2):
    counter = 0
    for s,p,o in G1:
        if (None,p,o) in G2:
            counter+=1
    return counter

def main():
    names=['Trumpington', 'Petersfield', 'None', "Queen Edith's", 'Newnham', 'Castle', 'East Chesterton', 'Abbey', 'Coleridge', 'Cherry Hinton', 'Market', 'West Chesterton', 'Arbury','Test']
    #trainEmbed()
    embedder_deepgraph = pickle.load(open("examples/embedder_deepgraph.pkl", "rb"))
    embedder_wl = pickle.load(open("examples/embedder_wl.pkl", "rb"))
    embedder_graph2vec = pickle.load(open("examples/embedder_graph2vec.pkl", "rb"))


    compare_dgk = KGCompare(embedder_deepgraph)
    compare_wl = KGCompare(embedder_wl)
    compare_g2v = KGCompare(embedder_graph2vec)
    m1 = compare_g2v.get_dist_matrix()
    m2 = compare_wl.get_dist_matrix()
    m3 = compare_dgk.get_dist_matrix()
    scaler = MinMaxScaler()

    # plt.plot(r[t])
    # plt.title('Distance to Graph 0 ' +method)
    # plt.xlabel('Graph ID')
    # plt.ylabel('Graph distance')
    # plt.show()
    graph_id=10

    norm1 = scaler.fit_transform(np.delete(m1[graph_id],graph_id).reshape(-1, 1))
    norm2 = scaler.fit_transform(np.delete(m2[graph_id],graph_id).reshape(-1, 1))
    norm3 = scaler.fit_transform(np.delete(m3[graph_id],graph_id).reshape(-1, 1))

    plt.plot(norm1, label='graph2vec')
    plt.plot(norm2, label='wl')
    plt.plot(norm3, label='dgk')


    plt.title('Scaled Distance to Graph ' +str(graph_id))
    plt.xlabel('Graph ID')
    plt.ylabel('Graph distance')
    plt.legend()
    plt.show()
    # start = time.time()
    # print(compare.get_dist_matrix())
    # stop = time.time()
    # print('time',stop-start)
    for i in range(len(embedder_graph2vec.graphs)):
         print(i, compare_g2v.get_most_similar(i),compare_g2v.get_least_similar(i))
         print(names[i], names[compare_g2v.get_most_similar(i)[0]],names[compare_g2v.get_least_similar(i)[0]])


    # visualize_distance_matrix(compare.get_dist_matrix())
    # visualize_distance_matrix(compare.get_graph_overlap())
    #
    # #compare based KG similarities
    # for i in range(len(embedder.graphs)):
    #     embed_dist = compare.get_dist_matrix()[i]
    #     graph_sims = compare.get_graph_overlap()[i]
    #     sorted_indexes=sorted(range(len(embed_dist)), key=lambda k: embed_dist[k])
    #     plt.xlabel('elements')
    #     plt.ylabel('distance')
    #     plt.plot(embed_dist[sorted_indexes],label='embeddings')
    #     plt.plot(1-graph_sims[sorted_indexes],label='graph overlap')
    #     plt.legend()
    #     plt.show()

if __name__ == '__main__':
    main()
