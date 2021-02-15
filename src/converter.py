import logging
import os

import rdflib
from rdflib.namespace import RDF
from rdflib import Graph

from tqdm import tqdm_notebook as tqdm



logger = logging.getLogger()
logger.setLevel("INFO")

#extracts the subgraph starting from a subject (in str). Does not extract any general TBOX information.
def extract_graph(subject,graph):
    current_result = rdflib.Graph()
    new_triples=set([])
    explored_individuals=set([])
    current_triple=rdflib.term.URIRef(subject)
    new_triples.add(current_triple)
    while  len(new_triples)>0:
        current_triple=new_triples.pop()
        for s,p,o in graph.triples( (current_triple, None, None) ):
            current_result.add((s,p,o))
            if o not in explored_individuals:
                new_triples.add(o)
        explored_individuals.add(current_triple)
    return current_result

def split_rdf_manually(working_folder, rdf_filename, splitting_query, rdf_file_format='xml'):

    logging.info('Reading RDF file manually...')
    graph = rdflib.Graph()
    graph.parse(rdf_filename, format=rdf_file_format)
    # Query what subjects have the given type:
    logging.info('Querying subjects that have the specified type...')
    subjects = graph.query(splitting_query)
        # Write an RDF file per subject:
    logging.info('Writing RDF files per subject...')
    for subject in tqdm(subjects):
        current_output_graph=extract_graph(subject[0],graph)
        print('The extrated graph contains %s percent of the triples from the original graph'%(str(len(current_output_graph)/len(graph))))
        logging.info('Writing the graph to file...')
        prefixed_subject = graph.namespace_manager.normalizeUri(subject[0])
        prefixed_subject=prefixed_subject.replace(':', '')
        if '<'in prefixed_subject and '/'in prefixed_subject:
            prefixed_subject=prefixed_subject[prefixed_subject.rfind('/')+1:-1]
        with open(os.path.join(working_folder, 'graphs_rdf', prefixed_subject.replace(':', '')+'_manually.xml'), 'wb') as current_output_file:
            current_output_graph.serialize(current_output_file)


#converts a set of graphs to vertices and edges with assigned IDs
#results in output_dir/graph.txt and the labels in output_dir/labels.txt
#the full graphs is necessary to extract the recurrent IDs in the subgraphs
def convert_to_gefsg(graphs,output_dir,split_file=False):
    #first make the ids for all the node labels and edge ids over the whole dataset
    edges_dict={}
    edges_dict_reverse={}
    label_dict={}
    label_dict_reverse={}
    label_id=0
    edge_id=0
    for graph in graphs:
        for subj, pred, obj in graph:
            if subj not in label_dict:
                label_dict[subj]=label_id
                label_dict_reverse[label_id]=subj
                label_id+=1
            if obj not in label_dict:
                label_dict[obj]=label_id
                label_dict_reverse[label_id]=obj
                label_id+=1
            if pred not in edges_dict:
                edges_dict[pred]=edge_id
                edges_dict_reverse[edge_id]=pred
                edge_id+=1
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    f= open(output_dir+"/labels.txt","w+")
    f.write('label ids\n')
    for i in range(0,len(label_dict)):
        f.write(str(i) + ' '+ label_dict_reverse[i]+'\n')
    f.write('edge ids\n')
    for i in range(0,len(edges_dict_reverse)):
        f.write(str(i) + ' '+edges_dict_reverse[i]+'\n')
    #convert the subgraphs based on the extracted labels
    f.write('subgraph ids\n')
    output_file=open(output_dir+"/graphs.txt","w+")
    for graphId,graph in enumerate(graphs):

        output_graph_file=output_file
        if split_file:
            output_graph_file=open(output_dir + "/" + str(graphId) + ".graph", "w+")
        #write the graph id to the labels file
        f.write(str(graphId) +'\n')
        output_graph_file.write('t # '+str(graphId)+'\n')
        nodes_dict={} #mapping nodes to ids
        nodes_dict_reverse={}
        node_id=0
        edges=[]
        for subj, pred, obj in graph:
            if subj not in nodes_dict:
                nodes_dict[subj] = node_id
                nodes_dict_reverse[node_id]=subj
                node_id += 1
            if obj not in nodes_dict:
                nodes_dict[obj] = node_id
                nodes_dict_reverse[node_id]=obj
                node_id += 1
            edges.append((nodes_dict[subj],nodes_dict[obj],edges_dict[pred]))

        for i in range(0,node_id):
            output_graph_file.write('v '+ str(i)+ ' '+ str(label_dict[nodes_dict_reverse[i]])+'\n')
        for i in range(0,len(edges)):
            output_graph_file.write('e %s %s %s\n'%(edges[i][0],edges[i][1],edges[i][2]))
        if split_file:
            output_graph_file.close()
    output_file.close()
    f.close()
    return label_dict, label_dict_reverse


# converts a set of graphs to vertices and edges with assigned IDs
# results in output_dir/graph.txt and the labels in output_dir/labels.txt
# the full graphs is necessary to extract the recurrent IDs in the subgraphs
def convert_to_json(full_graph_dir, list_sub_graphs_dirs, output_dir):
    # load the whole graph
    graph = Graph()
    graph.parse(location=full_graph_dir)
    # first make the ids for all the node labels and edge ids over the whole dataset
    edges_dict = {}
    edges_dict_reverse = {}
    label_dict = {}
    label_dict_reverse = {}
    label_id = 0
    edge_id = 0
    for subj, pred, obj in graph:
        if subj not in label_dict:
            label_dict[subj] = label_id
            label_dict_reverse[label_id] = subj
            label_id += 1
        if obj not in label_dict:
            label_dict[obj] = label_id
            label_dict_reverse[label_id] = obj
            label_id += 1
        if pred not in edges_dict:
            edges_dict[pred] = edge_id
            edges_dict_reverse[edge_id] = pred
            edge_id += 1
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    f = open(output_dir + "/labels.txt", "w+")
    f.write('label ids\n')
    for i in range(0, len(label_dict)):
        f.write(str(i) + ' ' + label_dict_reverse[i] + '\n')
    f.write('edge ids\n')
    for i in range(0, len(edges_dict_reverse)):
        f.write(str(i) + ' ' + edges_dict_reverse[i] + '\n')
    # convert the subgraphs based on the extracted labels
    graphId = 0
    f.write('subgraph ids\n')
    for subgraph_dir in list_sub_graphs_dirs:
        # create new graph file with id as file name
        output_file = open(output_dir + "/" + str(graphId) + ".json", "w+")

        subgaph = Graph()
        subgaph.parse(location=subgraph_dir)
        # write the graph id to the labels file
        f.write(str(graphId) + ' ' + subgraph_dir + '\n')

        graphId += 1
        nodes_dict = {}  # mapping nodes to ids
        nodes_dict_reverse = {}
        node_id = 0
        edges = []
        for subj, pred, obj in subgaph:
            if subj not in nodes_dict:
                nodes_dict[subj] = node_id
                nodes_dict_reverse[node_id] = subj
                node_id += 1
            if obj not in nodes_dict:
                nodes_dict[obj] = node_id
                nodes_dict_reverse[node_id] = obj
                node_id += 1
            edges.append((edges_dict[pred], nodes_dict[subj], nodes_dict[obj]))
        output_file.write('{')
        output_file.write('"edges": [')
        for i in range(0, len(edges)):
            output_file.write('[%s, %s]' % (edges[i][1], edges[i][2]))
            if i < len(edges) - 1:
                output_file.write(', ')
        output_file.write(']')

        output_file.write(', "features": {')
        for i in range(0, node_id):
            output_file.write('"%s": "%s" ' % (str(i), str(label_dict[nodes_dict_reverse[i]])))
            if i < node_id - 1:
                output_file.write(', ')
        output_file.write('}')
        output_file.write('}')
        output_file.close()
    f.close()

# Relables the graph such that edge labels are used as nodes
def convert_to_json_edge_relabling(list_sub_graphs, output_dir, label_dict={},
                                   label_dict_reverse={}):
    if not label_dict or not label_dict_reverse:
        print('Loading the graph')
        # first make the ids for all the node labels and edge ids over the whole dataset
        label_id = 0
        for sub_graph in list_sub_graphs:
            for subj, pred, obj in sub_graph:
                if subj not in label_dict:
                    label_dict[subj] = label_id
                    label_dict_reverse[label_id] = subj
                    label_id += 1
                if obj not in label_dict:
                    label_dict[obj] = label_id
                    label_dict_reverse[label_id] = obj
                    label_id += 1
                if pred not in label_dict:
                    label_dict[pred] = label_id
                    label_dict_reverse[label_id] = pred
                    label_id += 1
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            print('Writing the labels')
            f = open(output_dir + "/labels.txt", "w+")
            f.write('label ids\n')
            for i in range(0, len(label_dict)):
                f.write(str(i) + ' ' + label_dict_reverse[i] + '\n')
        f.close()
        print(label_dict)
    # convert the subgraphs based on the extracted labels and relable the edges
    graphId = 0
    for subgraph in list_sub_graphs:
        # create new graph file with id as file name
        output_file = open(output_dir + "/" + str(graphId) + ".json", "w+")



        # write the graph id to the labels file
        # f.write(str(graphId) + ' ' + subgraph_dir + '\n')

        graphId += 1
        nodes_dict = {}  # mapping nodes to ids
        nodes_dict_reverse = {}
        node_id = 0
        edge_dict = {}  # mapping nodes to ids
        edge_dict_reverse = {}
        edge_id = 0
        edges = []
        for subj, pred, obj in subgraph:
            if subj not in nodes_dict:
                nodes_dict[subj] = node_id
                nodes_dict_reverse[node_id] = subj
                node_id += 1
            if obj not in nodes_dict:
                nodes_dict[obj] = node_id
                nodes_dict_reverse[node_id] = obj
                node_id += 1
            if pred not in edge_dict:
                edge_dict[pred] = []
            edge_dict[pred].append(node_id)
            nodes_dict_reverse[node_id] = pred
            edges.append((nodes_dict[subj], node_id, nodes_dict[obj]))
            node_id += 1
        output_file.write('{')
        output_file.write('"edges": [')
        features = {}
        for i in range(0, len(edges)):
            output_file.write('[%s, %s], [%s, %s]' % (edges[i][0], edges[i][1], edges[i][1], edges[i][2]))
            if i < len(edges) - 1:
                output_file.write(', ')
        output_file.write(']')

        output_file.write(', "features": {')
        for i in range(0, node_id):
            output_file.write('"%s": "%s" ' % (str(i), str(label_dict[nodes_dict_reverse[i]])))
            if i < node_id - 1:
                output_file.write(', ')
        output_file.write('}')
        output_file.write('}')
        output_file.close()

    return label_dict, label_dict_reverse

def convert_to_json_edge_relabling_file_based(full_graph_dir, list_sub_graphs_dirs, output_dir, label_dict={},
                                   label_dict_reverse={}):
    if not label_dict or not label_dict_reverse:
        print('Loading the graph')
        # load the whole graph
        graph = Graph()
        graph.parse(location=full_graph_dir)
        # first make the ids for all the node labels and edge ids over the whole dataset
        label_id = 0
        for subj, pred, obj in graph:
            if subj not in label_dict:
                label_dict[subj] = label_id
                label_dict_reverse[label_id] = subj
                label_id += 1
            if obj not in label_dict:
                label_dict[obj] = label_id
                label_dict_reverse[label_id] = obj
                label_id += 1
            if pred not in label_dict:
                label_dict[pred] = label_id
                label_dict_reverse[label_id] = pred
                label_id += 1
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        print('Writing the labels')
        f = open(output_dir + "/labels.txt", "w+")
        f.write('label ids\n')
        for i in range(0, len(label_dict)):
            f.write(str(i) + ' ' + label_dict_reverse[i] + '\n')
        f.close()
        print(label_dict)
    # convert the subgraphs based on the extracted labels and relable the edges
    graphId = 0
    for subgraph_dir in list_sub_graphs_dirs:
        print('loading graph ' + subgraph_dir)
        # create new graph file with id as file name
        output_file = open(output_dir + "/" + str(graphId) + ".json", "w+")

        subgaph = Graph()
        subgaph.parse(location=subgraph_dir)

        # write the graph id to the labels file
        # f.write(str(graphId) + ' ' + subgraph_dir + '\n')

        graphId += 1
        nodes_dict = {}  # mapping nodes to ids
        nodes_dict_reverse = {}
        node_id = 0
        edge_dict = {}  # mapping nodes to ids
        edge_dict_reverse = {}
        edge_id = 0
        edges = []
        for subj, pred, obj in subgaph:
            if subj not in nodes_dict:
                nodes_dict[subj] = node_id
                nodes_dict_reverse[node_id] = subj
                node_id += 1
            if obj not in nodes_dict:
                nodes_dict[obj] = node_id
                nodes_dict_reverse[node_id] = obj
                node_id += 1
            if pred not in edge_dict:
                edge_dict[pred] = []
            edge_dict[pred].append(node_id)
            nodes_dict_reverse[node_id] = pred
            edges.append((nodes_dict[subj], node_id, nodes_dict[obj]))
            node_id += 1
        output_file.write('{')
        output_file.write('"edges": [')
        features = {}
        for i in range(0, len(edges)):
            output_file.write('[%s, %s], [%s, %s]' % (edges[i][0], edges[i][1], edges[i][1], edges[i][2]))
            if i < len(edges) - 1:
                output_file.write(', ')
        output_file.write(']')

        output_file.write(', "features": {')
        for i in range(0, node_id):
            output_file.write('"%s": "%s" ' % (str(i), str(label_dict[nodes_dict_reverse[i]])))
            if i < node_id - 1:
                output_file.write(', ')
        output_file.write('}')
        output_file.write('}')
        output_file.close()

    return label_dict, label_dict_reverse

# Relables the graph such that edge labels are used as nodes
def convert_to_dict(sub_graphs, label_dict={},
                                   label_dict_reverse={}):
    print(label_dict)
    print(label_dict_reverse)
    if not label_dict or not label_dict_reverse:
        print('Loading the graph')
        # load the whole graph
        for graph in sub_graphs:
            # first make the ids for all the node labels and edge ids over the whole dataset
            label_id = 0
            for subj, pred, obj in graph:
                if subj not in label_dict:
                    label_dict[subj] = label_id
                    label_dict_reverse[label_id] = subj
                    label_id += 1
                if obj not in label_dict:
                    label_dict[obj] = label_id
                    label_dict_reverse[label_id] = obj
                    label_id += 1
                if pred not in label_dict:
                    label_dict[pred] = label_id
                    label_dict_reverse[label_id] = pred
                    label_id += 1
    # convert the subgraphs based on the extracted labels and relable the edges
    graphId = -1
    graphs={}
    for subgaph in sub_graphs:
        # write the graph id to the labels file
        # f.write(str(graphId) + ' ' + subgraph_dir + '\n')

        graphId += 1
        nodes_dict = {}  # mapping nodes to ids
        nodes_dict_reverse = {}
        node_id = 0
        edge_dict = {}  # mapping nodes to ids
        edge_dict_reverse = {}
        edge_id = 0
        edges = []
        edge_node_dict={}
        for subj, pred, obj in subgaph:
            if subj not in nodes_dict:
                nodes_dict[subj] = node_id
                nodes_dict_reverse[node_id] = subj
                node_id += 1
            if obj not in nodes_dict:
                nodes_dict[obj] = node_id
                nodes_dict_reverse[node_id] = obj
                node_id += 1
            if pred not in edge_dict:
                edge_dict[pred] = []
            edge_dict[pred].append(node_id)
            nodes_dict_reverse[node_id] = pred
            edges.append((nodes_dict[subj], node_id, nodes_dict[obj]))
            if not nodes_dict[subj] in edge_node_dict:
                edge_node_dict[nodes_dict[subj]]=[]
            edge_node_dict[nodes_dict[subj]].append(nodes_dict[obj])
            if not nodes_dict[obj] in edge_node_dict:
                edge_node_dict[nodes_dict[obj]]=[]
            edge_node_dict[nodes_dict[obj]].append(nodes_dict[subj])
            node_id += 1
        graphs[graphId]={}
        for node in range(node_id):
            if node in edge_node_dict:
                graphs[graphId][node]={'neighbors':list(set(edge_node_dict[node])),'label':(label_dict[nodes_dict_reverse[node]],)}
            else:
                graphs[graphId][node] = {'neighbors': [], 'label': (label_dict[nodes_dict_reverse[node]],)}

    return graphs, label_dict_reverse

# print similarities
def get_similarity(graph1, graph2):
    sim_counter = 0
    for triple in graph1:
        if triple in graph2:
            sim_counter += 1
    return sim_counter / len(graph1)


def explore(subject, graph, depth, explored, results):
    for s, p, o in graph.triples((None, None, subject)):
        if not depth in results:
            results[depth] = []
        results[depth].append((s, p, o))
        if (s, p, o) not in explored:
            explored.append((s, p, o))
            explore(s, graph, depth + 1, explored, results)
        for s2, p2, o2 in graph.triples((subject, None, None)):
            if (s2, p2, o2) not in explored:
                explored.append((s2, p2, o2))
                explore(o2, graph, depth + 1, explored, results)

#converts graph1 gradually to graph2 and saves the changing graph to the output_dest.
#steps defines the number of intermediate graphs that will be saved.
def convert_graph_gradually(graph1, graph2, output_dest, steps):
    overlap = []
    for (s, p, o) in graph1:
        if (s, p, o) in graph2 and not 'http://www.w3.org/2000/01/rdf-schema' in str(p):
            overlap.append((s, p, o))
    if len(overlap) == 0:
        print('No overlap in graphs')
        return
    start_subj = overlap[0][0]
    results = {}
    if not os.path.exists(output_dest):
        os.mkdir(output_dest)
    explore(start_subj, graph1, 0, [], results)
    results_start = results
    results = {}
    explore(start_subj, graph2, 0, [], results)
    results_test = results
    if steps > len(results_start):
        print('#steps is larger than the possible changes')
        steps = len(results_start) - 1
    # note that test graph is deeper
    for i in range(0, len(results_start) - 1):
        start_index = len(results_start) - 1 - i
        remove_triples = results_start[start_index]
        add_triples = results_test[i]
        for remove_triple in remove_triples:
            graph1.remove(remove_triple)
        for add_triple in add_triples:
            graph1.add(add_triple)
        if i % steps == 0:
            graph1.serialize(destination=output_dest + '/d1_' + str(i) + '.xml', format='xml')
            print('sim new graph', get_similarity(graph1, graph2))

    # print('after')
    # for i in range(len(results_start),len(results_test)):
    #    add_triples=results_test[i]
    #    for add_triple in add_triples:
    #        start_graph.add(add_triple)
    #    start_graph.serialize(destination=output_dest+'/d1_'+str(i)+'.xml', format='xml')
    #    print('sim new graph',get_similarity(start_graph,test_graph))
# I added a second as something went wrond with Doc2Vec, however, I cant rember what i changes here...
def convert_graph_gradually2(graph1, graph2, output_dest, steps):
    ignoreP=['http://www.w3.org/2002/07/owl#disjointWith','http://www.w3.org/2000/01/rdf-schema#subClassOf']
    ignoreO=['http://www.w3.org/2002/07/owl#Class']
    overlap = []
    for (s, p, o) in graph1:
        if (None, p, o) in graph2 and str(p) not in ignoreP and str(o) not in ignoreO:
            overlap.append((s, p, o))
    if len(overlap) == 0:
        print('No overlap in graphs')
        return False
    start_subj = overlap[0][2]
    results = {}
    if not os.path.exists(output_dest):
        os.mkdir(output_dest)
    print(start_subj)
    explore(start_subj, graph1, 0, [], results)
    results_start = results
    results = {}
    explore(start_subj, graph2, 0, [], results)
    results_test = results
    if steps > len(results_start):
        print('#steps is larger than the possible changes')
        steps = len(results_start) - 1
    # note that test graph is deeper
    for i in range(0, len(results_start) - 1):
        if i < len(results_test):
            start_index = len(results_start) - 1 - i
            remove_triples = results_start[start_index]
            print('removing %s triples'%(len(remove_triples)))
            add_triples = results_test[i]
            for remove_triple in remove_triples:
                graph1.remove(remove_triple)
            print('Adding %s triples'%(len(add_triples)))

            for add_triple in add_triples:
                graph1.add(add_triple)
            if i % steps == 0:
                graph1.serialize(destination=output_dest + '/d1_' + str(i) + '.xml', format='xml')
                print('sim new graph', get_similarity(graph1, graph2))

#import glob
#covert to GraphML
#subgraphs_dir=glob.glob("../rdfdata/mutag/graphs_rdf/d*_manually.xml") #extract all the subgraphs dirs
#subgraphs_dir=['../rdfdata/mutag/graphs_rdf/d%s_manually.xml'%(str(i)) for i in range(len(subgraphs_dir))]

#convert_to_gefsg('../rdfdata/mutag/mutag.owl',subgraphs_dir,'../rdfdata/mutag/change_props_gefsg',split_file=True)
#convert_to_json_edge_relabling('../rdfdata/mutag/mutag.owl',subgraphs_dir,'../rdfdata/mutag/graphs_json2')