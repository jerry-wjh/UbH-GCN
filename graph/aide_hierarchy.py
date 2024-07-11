from audioop import reverse
import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 14

class Graph:
    def __init__(self, root=14, labeling_mode='spatial'):
        self.num_node = num_node
        self.root = root
        self.A = self.get_adjacency_matrix(labeling_mode)
        

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_hierarchical_graph(num_node, tools.get_edgeset(dataset='AIDE', root=self.root)) # L, 3, 25, 25
        else:
            raise ValueError()
        return A, self.root


if __name__ == '__main__':
    import tools
    g = Graph().A
    import matplotlib.pyplot as plt
    for i, g_ in enumerate(g[0]):
        plt.imshow(g_[1], cmap='gray')
        cb = plt.colorbar()
        plt.savefig('./graph_{}.png'.format(i))
        cb.remove()
        plt.show()