import numpy as np

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, hierarchy):
    A = []
    for i in range(len(hierarchy)):
        A.append(normalize_digraph(edge2mat(hierarchy[i], num_node)))

    A = np.stack(A)

    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_graph(num_node, edges):

    I = edge2mat(edges[0], num_node)
    Forward = normalize_digraph(edge2mat(edges[1], num_node))
    Reverse = normalize_digraph(edge2mat(edges[2], num_node))
    Forward_2 = normalize_digraph(k_adjacency(Forward, 2))
    Reverse_2 = normalize_digraph(k_adjacency(Reverse, 2))
    # A = np.stack((I, Forward, Reverse))
    A = np.stack((I, Forward, Reverse, Forward_2, Reverse_2))
    return A # 3(id, cp, cf), V, V

def get_hierarchical_graph(num_node, edges):
    A = []
    for edge in edges:
        A.append(get_graph(num_node, edge))
    A = np.stack(A)
    return A # len_edges, 3(id, cp, cf), V, V

def get_groups(dataset='AIDE', root=13):
    groups  =[]
        
    if dataset == 'AIDE':
        if root == 14:
            groups.append([14])
            groups.append([13])
            groups.append([1, 6, 7])
            groups.append([2, 3, 12, 8, 9])
            groups.append([4, 5, 10, 11])

        elif root == 1:
            groups.append([1])
            groups.append([2, 3, 12, 13])
            groups.append([4, 5, 14, 6, 7])
            groups.append([8, 9])
            groups.append([10, 11])

        else:
            raise ValueError()

    elif dataset == 'emilya':
        if root == 5:
            groups.append([5])
            groups.append([4,   6,  8,  12])
            groups.append([3,   7,  9,  13])
            groups.append([2,  16, 10,  14])
            groups.append([1,  11, 15])
            groups.append([17, 18])
        
        else:
            raise ValueError()

    return groups

def get_edgeset(dataset='AIDE', root=14):
    groups = get_groups(dataset=dataset, root=root)
    
    for i, group in enumerate(groups):
        group = [i - 1 for i in group]
        groups[i] = group

    identity = []
    forward_hierarchy = []
    reverse_hierarchy = []

    for i in range(len(groups) - 1):
        self_link = groups[i] + groups[i + 1]
        self_link = [(i, i) for i in self_link]
        identity.append(self_link)
        forward_g = []
        for j in groups[i]:
            for k in groups[i + 1]:
                forward_g.append((j, k))
        forward_hierarchy.append(forward_g)
        
        reverse_g = []
        for j in groups[-1 - i]:
            for k in groups[-2 - i]:
                reverse_g.append((j, k))
        reverse_hierarchy.append(reverse_g)

    edges = []
    for i in range(len(groups) - 1):
        edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])

    return edges