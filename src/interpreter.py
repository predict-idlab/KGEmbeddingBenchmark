from kg2vec import KGEmbedder
import glob
import numpy as np
from heapq import nlargest
import matplotlib.pyplot as plt
from gensim import utils, matutils

import networkx as nx
import pandas as pd
import pickle
# predict the words from a given embedding that has been trained with the same model
def predict_words_from_embedding(model, embedding, topn=10):
    # we use the internal weights of the doc2vec model to activate the words
    word_probs = np.dot(embedding, model.trainables.syn1neg.T)
    top_indices = matutils.argsort(word_probs, topn=topn, reverse=True)
    # returning the most probable output words with their probabilities
    return [(model.wv.index2word[index1], word_probs[index1]) for index1 in top_indices]


# converts WL paths back to graph paths
def extract_path_from_labels(word_list, all_labels):
    result = []
    for word in word_list:
        if word in all_labels:
            extracted = extract_path_from_labels(all_labels[word], all_labels)
            result.append(extracted)

        else:
            result.append(word)
    return result


# removes the prefix from the uri
def extract_prefix(uri):
    split_char = '#'
    if split_char not in uri:
        split_char = '/'
    return uri.split(split_char)[-1]


# converts graph paths back to uri's
def convert_label_to_uri(df_labels, path):
    converted = []
    for label in path:
        converted.append(extract_prefix(df_labels[df_labels.label == int(label)].ids.values.any()))
    return converted


# extracts the highest occuring individual in a set of paths
def get_highest_occurs(paths, df_labels):
    counter = 0
    individual_counter = {}
    for path in paths:
        converted = convert_label_to_uri(df_labels, path)
        for convert in set(converted):
            if not convert in individual_counter:
                individual_counter[convert] = 0
            individual_counter[convert] += 1
    highest = nlargest(50, individual_counter, key=individual_counter.get)
    results = {}
    for val in highest:
        results[val] = individual_counter.get(val)
    return results


def vis_graph(G, labels={}):
    nx.draw(G, with_labels=True, labels=labels)
    plt.figure(figsize=(40, 200))

    plt.show()
    # plt.savefig("graph.pdf")

def unfold_path(path,embedder,df_labels):

    def unfold_path_temp(path):
        extracted_path=embedder.all_new_labels[path]

        if extracted_path[0]  not in embedder.all_new_labels:
                #first order, extract label
            second_orders.append(path)
        else:
            for path_el in extracted_path:
                unfold_path_temp(path_el)
    second_orders=[]
    unfold_path_temp(path)
    G= nx.DiGraph()
    for second_order_node in second_orders:
        second_order_neighbours = embedder.all_new_labels[second_order_node]
        for neighbour in second_order_neighbours[1:]:
            G.add_edge(second_order_neighbours[0],neighbour)
    node_labels={}
    for node in list(G.nodes):
        node_labels[node]=extract_prefix(embedder.label_dict_reverse[int(node)])
    nx.set_node_attributes(G,  node_labels,name='labels')
    vis_graph(G,labels=node_labels)
    return G
def is_zero_order(path):
    try:#only the actual labels are numbers that can be parsed to strings. Faster than checking a list
        int(path)
        return True
    except:
        return False
def get_graphs_from_path(paths,df_labels,first_order,second_order,embedder):
    all_graphs=[]
    for path in paths:
        if is_zero_order(path):
            print(path,'zero order')
            G= nx.DiGraph()
            label=convert_label_to_uri(df_labels,[path])[0]
            G.add_node(label)
            vis_graph(G,labels={label:label})
            all_graphs.append(G)
        elif path in first_order:
            print(path,'first order')
            G= nx.DiGraph()
            for label in embedder.all_new_labels[path][1:]:
                G.add_edge(convert_label_to_uri(df_labels,embedder.all_new_labels[path][0])[0],convert_label_to_uri(df_labels,label)[0])
            node_labels={node:node for node in list(G.nodes)}

            vis_graph(G,labels=node_labels)
            all_graphs.append(G)

        elif path in second_order:
            print(path,'second order')
            G=unfold_path(path,embedder,df_labels)
            all_graphs.append(G)

        else:
            print(path,'nth order, must unfold')
            G=unfold_path(path)
            all_graphs.append(G)
    return all_graphs
def combine_graphs(all_graphs):
    G=nx.DiGraph()
    for graph in all_graphs:
        for u,v,data in graph.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if G.has_edge(u,v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
            G.nodes[u]['weight']  = G.nodes[u]['weight'] +1 if 'weight' in G.nodes[u] else 1.0
            G.nodes[v]['weight']  = G.nodes[v]['weight'] +1 if 'weight' in G.nodes[v] else 1.0

            node_labels={node:graph.nodes[node]['labels'] for node in graph.nodes() if 'labels' in graph.nodes[node]}
            nx.set_node_attributes(G,  node_labels,name='labels')
    labels={node:G.nodes[node]['labels'] for node in G.nodes() if 'labels' in G.nodes[node]}
    vis_graph(G,labels=labels)
    return G
def prune_combined_graph(G,n_threshold,e_threshold):
    G_selection=nx.DiGraph()

    for u,data in G.nodes(data=True):
        if data['weight'] >n_threshold:
            G_selection.add_node(u)
            for v,v_data in dict(G[u]).items():
                if v_data['weight'] > e_threshold:
                    G_selection.add_edge(u,v)

    new_labels={node:G.nodes[node]['labels'] for node in G_selection.nodes() if 'labels' in G.nodes[node]}
    nx.set_node_attributes(G,  new_labels,name='labels')

    vis_graph(G_selection,labels=new_labels)
    return G_selection
def diff_graphs(G1,G2):
    R=G1.copy()
    R.remove_nodes_from(n for n in G1 if n  in G2)
    new_labels={node:G1.nodes[node]['labels'] for node in R.nodes() if 'labels' in G1.nodes[node]}

    nx.set_node_attributes(R,  new_labels,name='labels')

    vis_graph(R,labels=new_labels)
    return R
def extract_path_orders(paths,embedder):
    zero_order = []
    first_order = []
    second_order = []
    for path in paths:
        if path not in embedder.all_new_labels:
            zero_order.append(path)
        else:
            extracted_paths = embedder.all_new_labels[path]
            for extracted_path in extracted_paths:
                if extracted_path not in embedder.all_new_labels:
                    first_order.append(path)
                    break
                else:
                    second_order.append(path)
                    break
    return zero_order,first_order,second_order

def main():
    #create a new embedder or load one from pickle
    embedder =  pickle.load(open( "examples/embedder.pkl", "rb" ))


    # we predict a certain number of words for an embedding
    num_word_predict = 100

    p_words = predict_words_from_embedding(embedder.model, embedder.model.docvecs[0], num_word_predict)
    p_words2 = predict_words_from_embedding(embedder.model, embedder.model.docvecs[1], num_word_predict)

    num_predict = 3
    paths = [p[0] for p in p_words[:num_predict]]
    paths2 = [p[0] for p in p_words2[:num_predict]]
    df_labels = pd.read_csv('examples/labels.txt', sep=' ')
    # identify which paths contain direct elements of the graph and which are the results of multiple WL passes
    zero_order,first_order,second_order=extract_path_orders(paths,embedder)
    zero_order2, first_order2, second_order2 = extract_path_orders(paths2, embedder)
    # retrieve the graph from the WL paths
    path_graphs = get_graphs_from_path(paths,df_labels,first_order,second_order,embedder)
    paths2_graphs = get_graphs_from_path(paths2,df_labels,first_order2,second_order2,embedder)
    #Combine all the paths together into one graph (for reconstruction)
    G_combined = combine_graphs(path_graphs)
    G2_combined = combine_graphs(paths2_graphs)
    #Lets prune the graph
    prune_combined_graph(G_combined, 20, 5)
    pruned = prune_combined_graph(G2_combined, 20, 5)
    #check the major differences
    R = diff_graphs(G_combined, G2_combined)
    R2 = diff_graphs(G2_combined, G_combined)
    #additional pruning is possible
    prune_combined_graph(R, 20, 8)
    prune_combined_graph(R2, 20, 8)

if __name__ == '__main__':
    main()