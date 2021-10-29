import dgl
from sklearn.neighbors import NearestNeighbors


def create_graph(pos, knn=None):
    u = []
    v = []

    if knn is None or len(pos)<knn:
        nbrs = NearestNeighbors(n_neighbors=len(pos)).fit(pos)
    else:
        nbrs = NearestNeighbors(n_neighbors=knn).fit(pos)

    distances, indices = nbrs.kneighbors(pos)
    for idx_list in indices:
        for k in idx_list[1:]:
            u.append(idx_list[0])
            v.append(k)

    graph = dgl.graph((v,u)) # Each node should RECEIVE 12 messages, so order is reverse
    return graph

class BatchSampler:
    def __init__(self, sizes, max_edges=1000000):
        self.samples = self.generate_samples(sizes, max_edges)
        self.idx = 0

    def generate_samples(self, sizes, max_edges):
        samples = []
        sample = []
        num_edges = 0
        for i, size in enumerate(sizes):
            if num_edges + size**2 > max_edges:
                samples.append(sample)
                sample = [i]
                num_edges = size**2
            else:
                num_edges += size**2
                sample.append(i)
        samples.append(sample)
        return samples

    def __iter__(self):
        return iter(self.samples)
