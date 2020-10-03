import networkx as nx
import numpy as np
from torch.utils.data import Dataset
import torch


class LineDataset(Dataset):

    def __init__(self, graph_file):
        super(LineDataset, self).__init__()
        self.g = nx.read_gpickle(graph_file)
        self.num_of_nodes = self.g.number_of_nodes()
        self.num_of_edges = self.g.number_of_edges()
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)

        try: # todo: when some edge does not have 'weight' attr -> error occur -> all edges are assigned weight of 1
            self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
        except:
            print('There is no weight attributes for edges')
            self.edge_distribution = np.ones(len(self.edges_raw), dtype=np.float32)
        else:
            print('G is weighted')

        #print('edge_distribution: ', list(self.edge_distribution)[:10])
        
        self.total_weight = np.sum(self.edge_distribution)
        self.edge_distribution /= self.total_weight
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        self.node_negative_distribution = np.power(
            np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

    def __len__(self):
        return self.num_of_edges

    def __getitem__(self, index, K=5):
        edge_index = self.edge_sampling.sampling()
        u_i = []
        u_j = []
        label = []

        edge = self.edges[edge_index]
        if self.g.__class__ == nx.Graph:
            if np.random.rand() > 0.5:      # important: second-order proximity is for directed edge
                edge = (edge[1], edge[0])
        u_i.append(edge[0])
        u_j.append(edge[1])
        label.append(1)
        for i in range(K):
            while True:
                negative_node = self.node_sampling.sampling()
                if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]]):
                    break
            u_i.append(edge[0])
            u_j.append(negative_node)
            label.append(-1)
        return u_i, u_j, label

    @staticmethod
    def collate(batches):
        source = [u for batch in batches for u in batch[0]]
        target = [v for batch in batches for v in batch[1]]
        label = [l for batch in batches for l in batch[2]]

        return source, target, torch.FloatTensor(label)

    def node_distribution_power(self, power):  # fuku
        node_distribution = np.power(np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), power)
        node_distribution /= np.sum(node_distribution)
        return node_distribution

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node, _ in self.nodes_raw}


class AliasSampling:

    # Reference: https://en.wikipedia.org/wiki/Alias_method

    #np.random.seed(42)

    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res

