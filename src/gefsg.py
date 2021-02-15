from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np
import sys

from utils.gSpan.gspan_mining import gSpan
from utils.gSpan.gspan_mining.config import parser
import glob
import os
import rdflib
import converter


class GEFSGEmbedder():
    def __init__(self,graphs,output_dir,min_sup=1,relable=True,clear_dir=False):
        self.graphs=graphs
        self.embeddings=[]
        self.model = None
        self.documents = None
        self.all_new_labels = None
        self.min_sup=min_sup
        self.output_dir = output_dir
        if clear_dir:
            self.clear_intermediate_files()
        if relable:
            self.label_dict, self.label_dict_reverse = converter.convert_to_gefsg(graphs, output_dir)
        doc_vec_params= {'vector_size':128,
                        'workers':4,
                        'alpha':0.0025,
                        'min_count':1,
                        'dm':0,
                        'epochs':100}
        self.set_doc_vec_params(doc_vec_params)
    def set_doc_vec_params(self,params):
        """set other doc_vec params"""
        self.max_epochs = params['epochs']
        del params['epochs']
        self.doc_vec_params=params

    def clear_intermediate_files(self):
        files = glob.glob(self.output_dir+'*')
        for f in files:
            os.remove(f)


    def train_embeddings(self,path, min_sup=2, min_path_len=3,max_path_len=3):
        args_str = '-s %s -d True -v False -l %s -u %s -p False -w True %s' % (min_sup, min_path_len,max_path_len, path)
        #args_str = '-s %s -d True -v False -p False -w True %s' % (min_sup, path)

        FLAGS, _ = parser.parse_known_args(args=args_str.split())
        documents = self.extract_frequenct_subraphs(FLAGS)
        # train the model
        model = Doc2Vec(documents,**self.doc_vec_params)
        return (model, documents)

    def embed(self,current_min_sup=None):
        #min_sup = 1
        # count the number of graphs to calc the min support based on the minimum relative support
        #min_sup = math.ceil(min_rel_sup * len(self.graphs))
        min_sup_temp = self.min_sup
        if current_min_sup:
            min_sup_temp = current_min_sup

        (model, document_collections) = self.train_embeddings(self.output_dir+'graphs.txt', min_sup_temp)
        results = []
        for i in range(len(model.docvecs)):
            results.append(model.docvecs[i])
        self.embeddings = np.array(results)
        return self.embeddings



    def extract_frequenct_subraphs(self,FLAGS):
        gs = gSpan(
                database_file_name=FLAGS.database_file_name,
                min_support=FLAGS.min_support,
                min_num_vertices=FLAGS.lower_bound_of_num_vertices,
                max_num_vertices=FLAGS.upper_bound_of_num_vertices,
                max_ngraphs=FLAGS.num_graphs,
                is_undirected=(not FLAGS.directed),
                verbose=FLAGS.verbose,
                visualize=FLAGS.plot,
                where=FLAGS.where
            )

        gs.run()
        gs.time_stats()
        # construct an array that indicates for each graph which subraphs are contained in it
        num_of_graphs=len(gs.graphs)
        graph_sub_graphs = [[] for y in range(num_of_graphs)]
        print(gs._report_df.columns)
        for index, row in gs._report_df.iterrows():
            for graph_id in row['test']:
                graph_sub_graphs[graph_id].append(str(index))
        # generate the document
        # the document is the mapping of each graph to the subraphs
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(graph_sub_graphs)]
        return documents



def test():
    import random
    from graphgen import GraphGen

    graphs_folder = '../../KGComparisonBenchmark/data/mutag/rdf_graphs/'
    graph_files = [graphs_folder+str(item)+'.ttl'
                               for item in range(len(glob.glob(graphs_folder + '/*.ttl')))
                               if os.path.isfile(graphs_folder+str(item)+'.ttl')]
    graphs=[rdflib.Graph().parse(graph_file, format="ttl") for graph_file in graph_files]
    selected_graph_files = random.sample(graph_files, 2)
    graph1 = rdflib.Graph()
    graph1.parse(selected_graph_files[0], format="ttl")
    graph2 = rdflib.Graph()
    graph2.parse(selected_graph_files[1], format="ttl")
    gen = GraphGen()
    gen.graph = graph1
    graphs = gen.convert_into(graph2, ignore_del=False)
    num_graphs=10
    selected_graphs = [graphs[i] for i in range(0,len(graphs),int(len(graphs)/num_graphs))]
    embedder = GEFSGEmbedder(selected_graphs,'example_gesfg/',min_sup=2,clear_dir=True)

    print(embedder.embed())

#test()