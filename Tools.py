import numpy as np

np.set_printoptions(threshold=np.nan)
import os
import pandas as pd
import random
from ToolsStructural import load_struct_data

ptnMAT_colab = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'netmats')
ptnMAT_d50_dir = os.path.join(ptnMAT_colab, '3T_HCP1200_MSMAll_d50_ts2')

ptnMAT_d50_ses1 = os.path.join(ptnMAT_d50_dir, 'netmats1.txt')
ptnMAT_d50_ses2 = os.path.join(ptnMAT_d50_dir, 'netmats2.txt')
pers_scores = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'TIVscores',
                           '1016_HCP_withTIV_acorrected_USETHIS.xlsx')
PTN_MAT_DIM = 50
"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
 
 adj - array of all adjacency matrices for all example graphs
 sizes - array of number of nodes for each graph
 nhood - the number of hops for the aggregation step: selecting the 'neighbours' 
"""


# transform a adjancy matrix with edge weights into a binary adj matrix (used for MASKED ATTENTION)
def get_binary_adj(graph):
    bin_adj = np.empty(graph.shape)
    for i in range(graph.shape[0]):
        for j in range(graph.shape[0]):
            if graph[i][j] > 0:
                bin_adj[i][j] = 1.0
    return bin_adj


def adj_to_bias(adj, sizes, nhood=1):
    # nr of graphs
    nb_graphs = adj.shape[0]
    # an empty matrix of the same shape as adj
    mt = np.empty(adj.shape)
    # iterate all the graphs
    for g in range(nb_graphs):
        # crate an identity matrix of the same shape as the adj for current graph (it includes only self-loops)
        mt[g] = np.eye(adj.shape[1])
        # create a adj matrix  to include nhood-hop neighbours
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (get_binary_adj(adj[g]) + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    # conjugate the adj matrix of nhood neighbours
    return -1e9 * (1.0 - mt)


def get_adj_ses1(dim):
    adj = []
    with open(ptnMAT_d50_ses1, 'r', encoding='UTF-8') as data:
        for line in data:
            graph = [[0 for x in range(dim)] for y in range(dim)]
            for index, edge_weight in enumerate(line.split()):
                graph[int(index / dim)][int(index % dim)] = float(edge_weight)
            adj.append(graph)

    return np.array(adj)


# get the NEO5 TIV scores as an array of length 5 vectors, one for each patient in the study
def get_NEO5_scores():
    df = pd.ExcelFile(pers_scores).parse('Raw_data')  # you could add index_col=0 if there's an index
    tiv_scores = []
    tiv_scores.append(df['NEO.NEOFAC_A'])
    tiv_scores.append(df['NEO.NEOFAC_O'])
    tiv_scores.append(df['NEO.NEOFAC_C'])
    tiv_scores.append(df['NEO.NEOFAC_N'])
    tiv_scores.append(df['NEO.NEOFAC_E'])

    return np.array(tiv_scores).transpose()


def gen_random_features(nb_nodes_graphs):
    features = []
    feat_vect = random.sample(range(1, 100), 10)
    for nb_nodes in nb_nodes_graphs:
        graph_feats = [feat_vect for node in range(nb_nodes)]
        features.append(graph_feats)
    return np.array(features)


def load_data():
    adj_matrices = get_adj_ses1(PTN_MAT_DIM)
    graphs_features = gen_random_features([adj.shape[0] for adj in adj_matrices])
    data_scores = get_NEO5_scores()[:adj_matrices.shape[0]]
    score_train, score_val, score_test = np.split(data_scores,
                                                  [int(len(data_scores) * 0.8), int(len(data_scores) * 0.9)])
    return adj_matrices, graphs_features, score_train, score_val, score_test


def shuffle_tr_data(scores, node_feats, bias_mats, adj_mats, chunk_sz):
    assert len(node_feats[:chunk_sz]) == chunk_sz
    shuffled_data = list(zip(scores,
                             node_feats[:chunk_sz],
                             bias_mats[:chunk_sz],
                             adj_mats[:chunk_sz]))

    random.shuffle(shuffled_data)
    score_in_tr, ftr_in_tr, bias_in_tr, adj_in_tr = map(np.array, list(zip(*shuffled_data)))
    assert len(score_in_tr) == len(ftr_in_tr) == len(bias_in_tr) == len(adj_in_tr) == chunk_sz

    return score_in_tr, ftr_in_tr, bias_in_tr, adj_in_tr


def shuffle_tr_utest(scores, node_feats, bias_mats, adj_mats, chunk_sz):
    for _ in range(50):
        score_in_tr, ftr_in_tr, bias_in_tr, adj_in_tr = shuffle_tr_data(scores, node_feats, bias_mats,
                                                                        adj_mats, chunk_sz)

        if score_in_tr.shape != score_train.shape: print("error")
        for row_s in score_in_tr:
            check = False
            for row in score_train:
                if set(row_s.tolist()) == set(row.tolist()): check = True
            if not check:
                print("error")
                break


if __name__ == "__main__":
    adj_matrices, graphs_features, score_train, score_test, score_val = load_struct_data()
    # used in order to implement MASKED ATTENTION by discardining non-neighbours out of nhood hops
    biases = adj_to_bias(adj_matrices, [graph.shape[0] for graph in adj_matrices], nhood=1)

    tr_size = len(score_train)
    adj = adj_matrices[:tr_size]
    feat = graphs_features[:tr_size]
    bias = biases[:tr_size]
    shuffle_tr_utest(score_train,feat,bias,adj,tr_size)