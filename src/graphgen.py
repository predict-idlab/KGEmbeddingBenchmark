
import random
import networkx as nx
import rdflib
from rdflib import URIRef
from rdflib.namespace import RDF, RDFS

import matplotlib.pyplot as plt
import copy
import uuid
random.seed(1337)


class GraphGen():
    def __init__(self):

        self.entity_counter = 0
        self.class_counter = 0
        self.property_counter = 0
        self.graph = rdflib.Graph()

    def add_random_triple(self):
        # adds a random triple
        # option1: generate a new entity or use an existing one
        # option2: generate a object property (a) or a type assertion (b)
        # option 2.a.1 generate a new object property or use an existing one
        # option 2.a.2 generate a new subject or use an existing one
        # option 2.b.1 generate a new class or use an existing on

        found_connection = False
        while not found_connection:
            org_ent_counter = self.entity_counter
            org_class_counter = self.class_counter
            org_prop_counter = self.property_counter

            enitity_index = random.randint(0, 2 * (self.entity_counter + 1))
            if enitity_index > self.entity_counter:
                self.entity_counter += 1
                enitity_index = self.entity_counter

            subject = URIRef("http://idlab.ugent.be/kgcompare/entity_" + str(self.entity_counter))
            # add subject or class?
            if random.random() > 0.5:
                class_index = random.randint(0, 2 * (self.class_counter + 1))
                if class_index > self.class_counter:
                    self.class_counter += 1
                    class_index = self.class_counter

                prop = RDF.type
                obj = URIRef("http://idlab.ugent.be/kgcompare/class_" + str(self.class_counter))
            else:
                enitity_index = random.randint(0, 2 * (self.entity_counter + 1))
                if enitity_index > self.entity_counter:
                    self.entity_counter += 1
                    enitity_index = self.entity_counter

                prop_index = random.randint(0, 2 * (self.property_counter + 1))
                if prop_index > self.property_counter:
                    self.property_counter += 1
                    prop_index = self.property_counter

                obj = URIRef("http://idlab.ugent.be/kgcompare/entity_" + str(self.entity_counter))
                prop = URIRef("http://idlab.ugent.be/kgcompare/prop_" + str(self.property_counter))
            if (any(self.graph.triples((subject, None, None))) or any(
                    self.graph.triples((None, None, obj)))) and not any(self.graph.triples((subject, prop, obj))):
                found_connection = True
                self.graph.add((subject, prop, obj))
            else:
                self.entity_counter = org_ent_counter
                self.class_counter = org_class_counter
                self.property_counter = org_prop_counter
        return self.graph

    def gen_random_rdf(self, nb_triples):
        # generates a random graph
        self.entity_counter = 0
        self.class_counter = 0
        self.property_counter = 0
        enity = URIRef("http://idlab.ugent.be/kgcompare/entity_" + str(self.entity_counter))
        clazz = URIRef("http://idlab.ugent.be/kgcompare/class_" + str(self.class_counter))
        self.graph.add((enity, RDF.type, clazz))
        for i in range(1, nb_triples):
            self.add_random_triple()
        return self.graph

    def convert_to_nx(self, relable=False,num_values=False,label_indexes = {},node_attributes=False):
        # converts the rdf graph to a networkx graph
        indexes = {}
        labels = {}
        def get_entity_label(entity):
            if entity not in label_indexes:
                label_indexes[entity] = len(label_indexes)
            return label_indexes[entity]
        def get_entity_id(entity):
            if not num_values:
                return str(entity)
            else:
                if not entity in indexes:
                    indexes[entity] = len(indexes)
                    #asign labels
                    if isinstance(entity,tuple):
                        #case (s,p,o)
                        labels[indexes[entity]]=get_entity_label(p)
                    else:
                        labels[indexes[entity]] = get_entity_label(entity)
                return indexes[entity]

        self.nx_indexes=indexes
        self.nx_labels = labels
        self.nx_label_indexes=label_indexes
        G = nx.Graph()

        for s, p, o in self.graph:
            if not relable:
                G.add_edge(get_entity_id(s), get_entity_id(o))
            else:
                G.add_edge(get_entity_id(s), get_entity_id((s, p, o)))
                G.add_edge(get_entity_id((s, p, o)), get_entity_id(o))
                # G.add_edge(get_entity_id(s,indexes), get_entity_id((s,p,o),indexes))
                # G.add_edge(get_entity_id((s,p,o),indexes), get_entity_id(o,indexes))
        if node_attributes:
            for i in range(len(G.nodes)):
                G.nodes[i]['label'] = self.nx_labels[i]
        return G

    def is_connected(self):
        # convert to networkx and check if undirected graph is connected
        G = self.convert_to_nx(relable=False)
        return nx.is_connected(G)

    def delete_random_triple(self):
        # deletes a random triple while making sure that the graph stays connected
        found = False
        (s, p, o) = (None, None, None)
        while not found and len(self.graph) > 1:
            (s, p, o) = random.choice(list(self.graph.triples((None, None, None))))
            self.graph.remove((s, p, o))
            if self.is_connected():
                found = True
            else:
                print('bad,', s, p, o)
                self.graph.add((s, p, o))
        return s, p, o

    def draw(self, relabling=False):
        G = self.convert_to_nx(relable=relabling)
        nx.draw(G, pos=nx.spring_layout(G))
        plt.show()

    def copy(self):
        return copy.deepcopy(self.graph)

    def convert_into(self,graph_into,ignore_add=False,ignore_del=False):
        g1_set = set(self.graph.triples((None, None, None)))
        g2_set = set(graph_into.triples((None, None, None)))
        g1_deletes = list(g1_set - g2_set)
        g1_adds = list(g2_set - g1_set)
        if ignore_add:
            g1_adds=[]
        if ignore_del:
            g1_deletes=[]
        graphs = []
        while len(g1_deletes) > 0 or len(g1_adds) > 0:
            # delete one
            found = False
            counter = 0
            while len(g1_deletes) > 0 and not found and counter < 100:
                to_rm_triple = random.choice(g1_deletes)

                self.graph.remove(to_rm_triple)
                if self.is_connected():
                    found = True
                    g1_deletes.remove(to_rm_triple)
                else:
                    self.graph.add(to_rm_triple)
                counter += 1
            # add one
            found = False
            counter = 0
            while len(g1_adds)>0 and not found and counter < 100:
                to_add_triple = random.choice(g1_adds)
                self.graph.add(to_add_triple)
                if self.is_connected():
                    found = True
                    g1_adds.remove(to_add_triple)
                else:
                    self.graph.remove(to_add_triple)
                counter += 1

            # store the graphs

            graphs.append(self.copy())
        return graphs
    def convert_hierarchical_schema(self,hierarchy_len=10):
        assigned_classes = set([o for s, p, o in self.graph.triples((None, RDF.type, None))])
        # create hierarchy
        hierarchy = {str(clazz): [str(clazz) + '_super_%s' % (counter) for counter in range(hierarchy_len)] for clazz in
                     assigned_classes}
        # add hierarchy to the ontology schema
        for clazz in assigned_classes:
            prev = clazz
            for hierarch in hierarchy[str(clazz)]:
                current = URIRef(hierarch)
                self.graph.add((prev, RDFS.subClassOf, current))
                prev = current
        graphs=[]
        for hierarchy_index in range(hierarchy_len):
            graph = self.copy()
            # retrieve assigned triples:
            for clazz in assigned_classes:
                remove_tripes = list(graph.triples((None, RDF.type, clazz)))
                new_class = URIRef(hierarchy[str(clazz)][hierarchy_index])
                add_triples = [(s, p, new_class) for (s, p, o) in remove_tripes]
                for remove_triple in remove_tripes: graph.remove(remove_triple)
                for add_triple in add_triples: graph.add(add_triple)
            graphs.append(graph)
        return graphs

    def convert_random_schema(self,num_graphs):
        assigned_classes = set([o for s, p, o in self.graph.triples((None, RDF.type, None))])
        graphs=[]
        for hierarchy_index in range(num_graphs):
            graph = self.copy()
            # retrieve assigned triples:
            for clazz in assigned_classes:
                remove_tripes = list(graph.triples((None, RDF.type, clazz)))
                new_class = URIRef(str(clazz)+str(uuid.uuid4()))
                add_triples = [(s, p, new_class) for (s, p, o) in remove_tripes]
                for remove_triple in remove_tripes: graph.remove(remove_triple)
                for add_triple in add_triples: graph.add(add_triple)
            graphs.append(graph)
        return graphs

    def convert_entities(self):
        assigned_entity = set([s for s, p, o in self.graph.triples((None, None, None))])
        graphs = []

        # retrieve assigned triples:
        for entity in assigned_entity:
            remove_tripes1 = list(self.graph.triples((entity, None, None)))
            remove_tripes2 = list(self.graph.triples((None, None, entity)))
            new_entity = URIRef(str(entity) + str(uuid.uuid4()))
            add_triples = [(new_entity, p, o) for (s, p, o) in remove_tripes1] + [(s, p, new_entity) for (s, p, o) in
                                                                                  remove_tripes2]
            for remove_triple in remove_tripes1 + remove_tripes2: self.graph.remove(remove_triple)
            for add_triple in add_triples: self.graph.add(add_triple)
            graphs.append(self.copy())
        return graphs

    def approximate_ged(self,graph1,graph2):
        '''
        Approximates the GED distance by counting the sum of the triples that should be removed from graph1 and added to convert it into graph2.
        '''
        g1_set = set(graph1.triples((None, None, None)))
        g2_set = set(graph2.triples((None, None, None)))
        g1_deletes = list(g1_set - g2_set)
        g1_adds = list(g2_set - g1_set)
        return len(g1_deletes) + len(g1_adds)

def main():
    # generate random RDF graphs
    nb_triples = 5
    gen = GraphGen()
    graph = gen.gen_random_rdf(100)
    graphs_rl = []
    graphs = []
    for i in range(30):
        gen.add_random_triple()
        G = gen.convert_to_nx(True)
        G.graph['gid'] = i
        graphs_rl.append(G)
        G2 = gen.convert_to_nx(False)
        G2.graph['gid'] = i
        graphs.append(G2)

    gen = GraphGen()
    graph = gen.gen_random_rdf(10)
    gen2 = GraphGen()
    graph2 = gen2.gen_random_rdf(10)
    gen.draw()

    graphs=gen.convert_into(graph2)
    for graph in graphs:
        counter=0
        for s,p,o in graph:
            if (s,p,o) in graphs[0]:
                counter+=1
        print('number of equal triples:',counter)


def schema_test():
    import glob
    import os
    graphs_folder = '../KGComparisonBenchmark/data/mutag/rdf_graphs/'
    graph_files = [graphs_folder + str(item) + '.ttl'
                   for item in range(len(glob.glob(graphs_folder + '/*.ttl')))
                   if os.path.isfile(graphs_folder + str(item) + '.ttl')]
    num_iterations = 2
    # randomly select two graphs
    random.seed(1337)

    selected_graph_files = random.sample(graph_files, 2)
    graph1 = rdflib.Graph()
    graph1.parse(selected_graph_files[0], format="ttl")
    graph2 = rdflib.Graph()
    graph2.parse(selected_graph_files[1], format="ttl")
    gen = GraphGen()
    gen.graph = graph1
    hierarch_graphs = gen.convert_hierarchical_schema()
    random_graphs = gen.convert_random_schema(num_graphs=10)
    for s,p,o in random_graphs[0]:
        print(s,p,o)

    entity_graphs = gen.convert_entities()

    for s,p,o in entity_graphs[0]:
        print(s,p,o)


    for graph in entity_graphs:
        print(gen.approximate_ged(entity_graphs[0],graph))