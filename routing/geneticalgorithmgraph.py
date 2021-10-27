from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from haversine import haversine
from numba import njit, prange
from scipy.spatial.distance import cdist

from abstractgraph import AbstractGraphClass

# Parallel computation utilities

@njit(parallel=True, fastmath=True)
def fast_sum(route, dimensions, dist_matrix):
    total_distance = 0.0
    for i in prange(dimensions):
        total_distance += dist_matrix[route[i], route[i+1]]
    return total_distance

@njit()
def fast_breed(size, mating_pool, dimensions, start_node,
               end_node, nodes, dist_matrix):
    children = np.zeros(shape=(size, mating_pool.shape[1]))
    for i in prange(size):
        parents = mating_pool[np.random.choice(
            np.arange(mating_pool.shape[0]), size=2, replace=False)]
        parent_1 = parents[0]
        parent_2 = parents[1]
        gene = np.random.randint(low=1, high=dimensions+1)
        children[i, 1:gene] = parent_1[1:gene]
        children[i, gene:-2] = np.array([node for node in parent_2[1:-2]
                                         if not np.any(node == parent_1[1:gene])])
        children[i, -2] = 0 if np.all(end_node ==
                                      start_node) else nodes[-1]
        children[i, -1] = fast_sum(children[i].astype(np.int32),
                                   dimensions+1, dist_matrix)
    return children

@njit()
def fast_mutate(individuals, dimensions, dist_matrix):
    mutated = individuals.copy()
    for i in prange(individuals.shape[0]):
        genes = np.random.choice(
            np.arange(1, dimensions+1), size=2, replace=False)
        gene_1 = genes[0]
        gene_2 = genes[1]
        mutated[i, gene_1], mutated[i, gene_2] = mutated[i,
                                                         gene_2], mutated[i, gene_1]
        mutated[i, -1] = fast_sum(mutated[i].astype(np.int32),
                                  dimensions+1, dist_matrix)
    return mutated

@njit()
def fast_2_opt(route, size, dimensions, dist_matrix):
    results = np.zeros(shape=(size, route.shape[0]+1))
    for i in prange(size):
        nodes = np.random.choice(
            np.arange(1, dimensions+1), size=2, replace=False)
        j, k = np.sort(nodes)
        results[i, :-1] = route.copy()
        results[i, j:k+1] = results[i, j:k+1][::-1].copy()
        results[i, -1] = fast_sum(results[i].astype(np.int32),
                                  dimensions+1, dist_matrix)
    results = results[results[:, -1].argsort()]
    return results[0]


'''
Genetic Algorithm to find an optimal solution to a routing problem.
'''
class GeneticAlgorithmGraph(AbstractGraphClass):

    def __init__(self, data, start_node, end_node=None, **kwargs):
        self.pop_size = kwargs['pop_size']
        self.elite_size = kwargs['elite_size']
        self.mutation_rate = kwargs['mutation_rate']
        self.generations = kwargs['generations']
        self.metric = 'euclidean'
        if kwargs.get('metric') == 'km':
            self.metric = haversine
        self.start_node = np.asarray(start_node)
        self.end_node = np.asarray(end_node) if end_node else self.start_node
        # aggiungi il deposito come primo nodo se i nodi iniziale e finale coincidono
        # altrimenti, aggiungi il nodo di partenza in cima e quello di arrivo in fondo
        if np.all(self.end_node != self.start_node):
            extended_data = np.concatenate(
                ([self.start_node], data, [self.end_node]), axis=0)
            self.dimensions = extended_data.shape[0] - 2
        else:
            extended_data = np.concatenate(([self.start_node], data), axis=0)
            self.dimensions = extended_data.shape[0] - 1
        # posizioni geografiche dei nodi
        self.positions = extended_data
        self.compute_distances(extended_data)
        # etichette dei nodi nel grafo
        self.nodes = np.arange(extended_data.shape[0])
        self.population = np.empty(shape=(self.pop_size, self.dimensions+3))
        self.shortest_path_length = None
        self.shortest_path = None

    def __del__(self):
        del (self.nodes)
        del (self.population)
        del (self.dimensions)
        del (self.positions)
        del (self.dist_matrix)
        del (self.shortest_path)

    def compute_distances(self, data):
        self.dist_matrix = ((cdist(data, data, self.metric)
                             * 100000).astype(int)/100000).astype(float)
    
    def route_length(self, route):
        return fast_sum(route.astype(int), self.dimensions+1, self.dist_matrix)

    def create_route(self):
        route = np.zeros(self.dimensions+3)
        route[1:-2] = np.random.choice(range(1, self.dimensions+1),
                                       size=self.dimensions, replace=False)
        route[-2] = 0 if np.all(self.end_node ==
                                self.start_node) else self.nodes[-1]
        route[-1] = self.route_length(route)
        return route

    def generate_initial_population(self):
        self.population[:] = [self.create_route()
                              for i in range(self.pop_size)]
    def rank_routes(self):
        self.population = self.population[self.population[:, -1].argsort()]

    def select_routes(self):
        selection = self.population.copy()
        selection[self.elite_size:] = self.population[np.random.randint(
            low=0, high=self.pop_size, size=self.pop_size-self.elite_size)]
        selection = selection[selection[:, -1].argsort()]
        return selection

    def generate_mating_pool(self):
        self.rank_routes()
        mating_pool = self.select_routes()
        return mating_pool

    def breed_population(self, mating_pool):
        self.population[:self.elite_size] = mating_pool[:self.elite_size]
        self.population[self.elite_size:] = fast_breed(
            self.pop_size-self.elite_size, mating_pool, self.dimensions, self.start_node,
            self.end_node, self.nodes, self.dist_matrix)

    def mutate_population(self):
        chances = np.concatenate((np.ones(
            shape=self.elite_size), np.random.random_sample(size=self.pop_size-self.elite_size)), axis=0)
        if np.any(chances < self.mutation_rate):
            to_mutate = chances < self.mutation_rate
            self.population[to_mutate] = fast_mutate(
                self.population[to_mutate], self.dimensions, self.dist_matrix)

    def get_next_generation(self):
        mating_pool = self.generate_mating_pool()
        self.breed_population(mating_pool)
        self.mutate_population()
        self.rank_routes()

    def reverse_3_opt(self, route, i, j, k, l):
        outcomes = np.repeat([np.concatenate((route, [0.0]))], 8, axis=0)
        for outcome, flip in enumerate(product([0, 1], repeat=3)):
            if outcome > 0:
                if flip[0]:
                    outcomes[outcome, i:j +
                             1] = np.flip(outcomes[outcome, i:j + 1], axis=0).copy()
                if flip[1]:
                    outcomes[outcome, j+1:k +
                             1] = np.flip(outcomes[outcome, j+1:k + 1], axis=0).copy()
                if flip[2]:
                    outcomes[outcome, k:l +
                             1] = np.flip(outcomes[outcome, k:l + 1], axis=0).copy()
            outcomes[outcome, -1] = self.route_length(outcomes[outcome])
        return outcomes[outcomes[:, -1].argsort()][0]

    def apply_2_opt(self):
        no_improvement = 0
        while no_improvement < 1e4:
            no_improvement += 1
            path = fast_2_opt(self.shortest_path, 50,
                              self.dimensions, self.dist_matrix)
            cost = path[-1]
            if cost < self.shortest_path_length:
                no_improvement = 0
                self.shortest_path = path[:-1].astype(int)
                self.shortest_path_length = cost

    def apply_3_opt(self):
        no_improvement = 0
        while no_improvement < 1e3:
            no_improvement += 1
            i, j, k, l = np.sort(np.random.choice(
                np.arange(1, self.dimensions+1), size=4, replace=None))
            path = self.reverse_3_opt(
                self.shortest_path, i, j, k, l)
            cost = path[-1]
            if cost < self.shortest_path_length:
                no_improvement = 0
                self.shortest_path = path[:-1].astype(int)
                self.shortest_path_length = cost

    def get_shortest_path(self):
        self.history = np.zeros(shape=(self.generations, 2))
        self.generate_initial_population()
        for i in range(self.generations):
            self.get_next_generation()
            self.history[i] = [i+1, self.population[0, -1]]
        self.shortest_path_length = self.population[0, -1]
        self.shortest_path = self.population[0, :-1].astype(int)
        self.apply_2_opt()
        self.apply_3_opt()
        return self.shortest_path

    def get_shortest_path_length(self):
        if self.shortest_path_length is None:
            self.get_shortest_path()
        return self.shortest_path_length

    def draw_path(self, filename=None, figsize=(8, 8), with_labels=True, width=0.5, node_size=300, alpha=0.7, font_size=8):
        graph = nx.DiGraph()
        for i in range(len(self.shortest_path[:-1])):
            graph.add_edge(self.shortest_path[i], self.shortest_path[i+1])

        G = nx.path_graph(self.shortest_path, create_using=graph)
        positions = {}
        val_map = {0: 'r'}
        values = [val_map.get(node, 'c') for node in G.nodes()]
        for node in self.shortest_path[:-1]:
            positions[node] = self.positions[node]

        plt.figure(figsize=figsize)
        nx.draw_networkx(G, pos=positions, with_labels=with_labels, width=width, node_size=node_size,
                         node_color=values, alpha=alpha, font_size=font_size)
        if filename:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()  # display
        plt.close()
