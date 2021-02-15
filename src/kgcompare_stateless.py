import numpy as np
from scipy import spatial
from rdflib import Graph


class StatelessKGCompare:
    """
    Stateless KG Compare helper class
    """
    @staticmethod
    def get_dist_matrix(embeddings, distance_metric=spatial.distance.cosine):
        """computes distances matrix based on the embedding in the KGEmbedder"""
        all_distances=[]
        for embedi in embeddings:
            distances=[]
            for embedj in embeddings:
                distances.append(StatelessKGCompare.calc_distance(embedi,embedj, distance_metric))
            all_distances.append(np.array(distances))
        return np.array(all_distances)

    @staticmethod
    def calc_distance(embedding1, embedding2, distance_metric=spatial.distance.cosine):
        return distance_metric(embedding1, embedding2)

    @staticmethod
    def get_generic_similarity(graph_id, embeddings, similarity_measure, distance_metric=spatial.distance.cosine):
        """returns list with most similar graph ids, based on generic similarity measure, either min/max"""
        matrix = StatelessKGCompare.get_dist_matrix(embeddings, distance_metric)
        min_dist = similarity_measure(np.concatenate([matrix[graph_id][:graph_id], matrix[graph_id][graph_id+1:]]))
        return [i for i, j in enumerate(matrix[graph_id]) if j == min_dist]

    @staticmethod
    def get_most_similar(graph_id, embeddings, distance_metric=spatial.distance.cosine):
        """returns list with most similar graph ids"""
        return StatelessKGCompare.get_generic_similarity(graph_id, embeddings, min, distance_metric)

    @staticmethod
    def get_least_similar(graph_id, embeddings, distance_metric=spatial.distance.cosine):
        """returns list with least similar graph ids"""
        return StatelessKGCompare.get_generic_similarity(graph_id, embeddings, max, distance_metric)

    @staticmethod
    def get_graph_overlap(graphs):
        return np.array([np.array([po_match(G1,G2)/len(G1) for G1 in graphs]) for G2 in graphs])


def po_match(G1: Graph, G2: Graph):
    counter = 0
    for s,p,o in G1:
        if (None,p,o) in G2:
            counter+=1
    return counter
