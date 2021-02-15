"""SimGNN runner."""

from utils.simgnn.utils import tab_printer
from utils.simgnn.simgnn import SimGNNTrainer
from utils.simgnn.param_parser import parameter_parser
import pickle
import json
import networkx
def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    trainer.fit()
    trainer.score()

def networkx_to_json(g1, g2, ged):
    G1 = networkx.read_gexf('/Users/psbonte/Documents/Github/SimGNN/dataset/gexf/IMDBMulti/train/'+str(g1)+'.gexf')
    G2 = networkx.read_gexf('/Users/psbonte/Documents/Github/SimGNN/dataset/gexf/IMDBMulti/train/'+str(g2)+'.gexf')

    json = {}
    json['graph_1'] = [[int(e[0]), int(e[1])] for e in list(G1.edges)]
    json['graph_2'] = [[int(e[0]), int(e[1])] for e in list(G2.edges)]
    labels1 = networkx.get_node_attributes(G1, 'label')
    json['labels_1'] = [labels1[str(i)] for i in range(len(list(G1.nodes)))]
    labels2 = networkx.get_node_attributes(G2, 'label')
    json['labels_2'] = [labels2[str(i)] for i in range(len(list(G2.nodes)))]
    json['ged'] = ged
    return json

def convert_gexf():
    infile = open(
        '/Users/psbonte/Documents/Github/SimGNN/dataset/gexf/IMDBMulti/imdbmulti_ged_astar_gidpair_dist_map.pickle',
        'rb')
    new_dict = pickle.load(infile)
    infile.close()
    graph_counter=0
    for graph_pair in new_dict:
        if graph_pair[0]<100 and graph_pair[1]<100:
            try:
                print('converting',graph_pair)
                graph_temp = networkx_to_json(graph_pair[0], graph_pair[1], new_dict[graph_pair])
                print('writing to file')
                with open('/Users/psbonte/Documents/Github/SimGNN/dataset/gexf/IMDBMulti/json/train/'+str(graph_counter)+'.json', 'w') as outfile:
                    json.dump(graph_temp, outfile)
                graph_counter += 1
                print('done writing')
            except:
                print('file not found',graph_pair)

if __name__ == "__main__":
    #convert_gexf()

    main()
