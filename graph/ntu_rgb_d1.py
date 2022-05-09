import sys

sys.path.extend(['../'])
from graph import tools

num_node = 36
self_link = [(i, i) for i in range(num_node)]
inward= [(1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),
                    (8, 1), (9, 8), (10, 9), (11, 1), (12, 11), (13, 12),
                    (14, 0), (15, 0), (16, 14), (17, 15), (19, 18), (20, 19),
                    (21, 20), (22, 21), (23, 19), (24, 23), (25, 24),
                    (26, 19), (27, 26), (28, 27), (29, 19), (30, 29),
                    (31, 30), (32, 18), (33, 18), (34, 32), (35, 33)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()

    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
