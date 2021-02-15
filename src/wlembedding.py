from graph2vec import feature_extractor,save_embedding
import converter
import glob
from rdflib import Graph
import time
import pickle
import os
class WLEmbedder:
    def __init__(self, graphs, output_dir,clear_dir=False):
        self.graphs = graphs
        self.embeddings = []
        self.output_dir = output_dir
        if clear_dir:
            self.clear_intermediate_files()
        self.label_dict, self.label_dict_reverse = converter.convert_to_json_edge_relabling(graphs, output_dir,label_dict={},label_dict_reverse={})
        subgraphs_dir = glob.glob(output_dir + '*.json')
        self.json_graphs = [output_dir+'/%s.json' % str(i) for i in range(len(subgraphs_dir))]

    def clear_intermediate_files(self):
        files = glob.glob(self.output_dir+'*')
        for f in files:
            os.remove(f)
    def gen_wl_vectors(self, depth):
        document_collections = []
        all_features = set([])
        all_new_labels = {}
        document_collections = []

        # extract wl features
        for g in self.json_graphs:
            doc, new_labels = feature_extractor(g, depth)
            document_collections.append(doc)
            all_new_labels.update(new_labels)
            all_features.update(document_collections[-1].words)
        # extract unique features and sort them (original features first)
        all_features = list(all_features)
        # normally the original node labels come first in the vector, in our case it's not important
        # all_features.sort(key=len, reverse=False)
        # count the number of occurences of each feature in each graph
        wl_vectors = []
        for doc in document_collections:
            wl_vec = []
            for feature in all_features:
                wl_vec.append(doc.words.count(feature))
            wl_vectors.append(wl_vec)
        # the WL vector is a count of each feature in each graph
        return wl_vectors, all_new_labels, all_features

    def embed(self,depth=2):
        wl_vectors, all_new_labels,all_features=self.gen_wl_vectors(depth)
        self.embeddings=wl_vectors
        return wl_vectors

def main():
    graph_locs = '/Users/psbonte/Documents/Projects/Radiance/Data/ML6/Cambridge/context/graphs/'
    names=['Trumpington', 'Petersfield', 'None', "Queen Edith's", 'Newnham', 'Castle', 'East Chesterton', 'Abbey', 'Coleridge', 'Cherry Hinton', 'Market', 'West Chesterton', 'Arbury','Test']
    graphs = []
    for graph_name in names:
        g = Graph()
        g.parse(graph_locs+graph_name+".ttl", format="ttl")
        graphs.append(g)

    start = time.time()
    embedder = WLEmbedder(graphs,'examples/')

    embeddings = embedder.embed()
    end = time.time()

    print(embeddings)
    print('time to embed',end - start)
    pickle.dump(embedder,open( "examples/embedder_wl.pkl", "wb" ))

if __name__ == '__main__':
    main()