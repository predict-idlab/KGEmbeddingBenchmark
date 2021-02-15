import os

import converter
from graph2vec import feature_extractor,save_embedding
from rdflib import Graph
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import glob
import numpy as np
import pickle
import time
import os
from natsort import natsorted

class KGEmbedder:
    def __init__(self,graphs,output_dir,relable=True,clear_dir=False,hyperparameters={'vector_size':128,
                                                                                      'alpha':0.025,
                                                                                      'min_alpha':0.00025,
                                                                                      'min_count':1,'dm':0,
                                                                                      'epochs':100,
                                                                                      'alpha_decay': 0.0002,
                                                                                      'wl_rounds': 4}):
        self.graphs=graphs
        self.embeddings=[]
        self.model = None
        self.documents = None
        self.all_new_labels = None

        self.output_dir = output_dir
        if clear_dir:
            self.clear_intermediate_files()
        if relable:
            self.label_dict, self.label_dict_reverse = converter.convert_to_json_edge_relabling(graphs, output_dir,label_dict={},
                                   label_dict_reverse={})
        self.json_graphs = natsorted(glob.glob(output_dir+'*.json'))
        print(self.json_graphs)
        self.hyperparameters = hyperparameters
        self.set_doc_vec_params({
            param: val
            for param, val
            in hyperparameters.items()
            if param not in ['alpha_decay', 'wl_rounds']
        })

    def set_doc_vec_params(self,params):
        """set other doc_vec params"""
        self.max_epochs = params['epochs']
        del params['epochs']
        self.doc_vec_params=params

    def clear_intermediate_files(self):
        files = glob.glob(self.output_dir+'*')
        for f in files:
            os.remove(f)

    def train_embeddings(self,output_dest=None):
        """Starts embedding training"""
        print("\nFeature extraction started.\n")

        all_new_labels={}
        document_collections= []
        counter=0
        for g in self.json_graphs:
            doc,new_labels=feature_extractor(g, self.hyperparameters['wl_rounds'],fixed_name=str(counter))
            document_collections.append(doc)
            all_new_labels.update(new_labels)
            counter+=1

        print("\nOptimization started.\n")


        model = Doc2Vec(**self.doc_vec_params)

        model.build_vocab(document_collections)

        for epoch in range(self.max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(document_collections,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= self.hyperparameters['alpha_decay']
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        if not output_dest == None:
            save_embedding(output_dest, model, self.json_graphs, self.doc_vec_params['vector_size'])
        self.model=model
        self.documents =document_collections
        self.all_new_labels = all_new_labels
        return (model,document_collections,all_new_labels)

    def embed(self):
        """Computes and returns embeddings"""
        (model,document_collections,_)=self.train_embeddings()
        results=[]
        for i in range(len(model.docvecs)):
            results.append(model.docvecs[i])
        self.embeddings=np.array(results)
        return self.embeddings

    def get_embedding(self,graph,output_dir):
        label_dict, label_dict_reverse = converter.convert_to_json_edge_relabling([graph], output_dir,
                                                                                            label_dict={},
                                                                                            label_dict_reverse={})
        json_graphs = natsorted(glob.glob(output_dir + '*.json'))
        counter=0
        document_collections = []
        for g in json_graphs:
            doc,new_labels=feature_extractor(g, self.hyperparameters['wl_rounds'],fixed_name=str(counter))
            document_collections.append(doc)
            counter+=1
        return self.model.infer_vector(document_collections[0].words)

def main2():
    graph_locs = '/Users/psbonte/Documents/Projects/Radiance/Data/ML6/Cambridge/context/graphs/'
    names=['Trumpington', 'Petersfield', 'None', "Queen Edith's", 'Newnham', 'Castle', 'East Chesterton', 'Abbey', 'Coleridge', 'Cherry Hinton', 'Market', 'West Chesterton', 'Arbury','Test']
    graphs = []
    for graph_name in names:
        g = Graph()
        g.parse(graph_locs+graph_name+".ttl", format="ttl")
        graphs.append(g)

    start = time.time()
    embedder = KGEmbedder(graphs,'examples/')

    embeddings = embedder.embed()
    end = time.time()

    print(embeddings)
    print('time to embed',end - start)
    pickle.dump(embedder,open( "examples/embedder_graph2vec.pkl", "wb" ))

def main():
    graph_locs = '/Users/psbonte/Documents/Projects/Radiance/Data/Skyline/radiance_skyline/data/graphs/'
    #names=['GENT_1000', 'GENT_1007', 'GENT_1010', 'GENT_1028', 'GENT_1033', 'GENT_1038', 'GENT_1044', 'GENT_1056', 'GENT_106', 'GENT_1069', 'GENT_1074', 'GENT_1126', 'GENT_1132', 'GENT_1139', 'GENT_114', 'GENT_1142', 'GENT_1161', 'GENT_1168', 'GENT_117', 'GENT_1170', 'GENT_1193', 'GENT_1201', 'GENT_1206', 'GENT_1216', 'GENT_124', 'GENT_1241', 'GENT_1278', 'GENT_1296', 'GENT_1311', 'GENT_1327', 'GENT_1334', 'GENT_1335', ]
    names = natsorted(glob.glob(graph_locs+'*.ttl'))
    results_all=[]
    results_avg = []
    time_predict = []
    for num_graphs in [10,50,100,200,500,1000]:
        names_current = names[:num_graphs]
        graphs = []
        for graph_name in names_current:
            g = Graph()
            g.parse(graph_name, format="ttl")
            graphs.append(g)

        start = time.time()
        embedder = KGEmbedder(graphs,'examples/',clear_dir=True)

        embeddings = embedder.embed()
        end = time.time()

        print(embeddings)
        print('time to embed',end - start)
        results_all.append(end-start)
        print('time to embed single graph', (end - start)/len(graphs))
        results_avg.append((end - start)/len(graphs))
        start = time.time()
        embedder.get_embedding(graphs[0],'test_test/')
        end = time.time()
        time_predict.append(end - start)
        print(embeddings)
        print('time to predict', end - start)
    print(results_all)
    print(results_avg)
    print(time_predict)

    #pickle.dump(embedder,open( "examples/embedder_graph2vec.pkl", "wb" ))
if __name__ == '__main__':
    main()