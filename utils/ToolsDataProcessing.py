import numpy as np
import os
import pandas as pd
import random
import sys
from itertools import product

# Excel file containing per-subject personality scores
pers_scores = os.path.join(os.pardir, 'Data', 'TIVscores', '1016_HCP_TIVscores.xlsx')


################### util functions for processing PERSONALITY SCORES ########################

# dictionary of string subject ID: array of real-valued scores for each trait
def get_NEO5_scores(trait_choice: list = None) -> tuple:
    df = pd.ExcelFile(pers_scores).parse('Raw_data')
    tiv_scores = []
    if trait_choice is None:
        trait_names = ['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E']
    else:
        trait_names = trait_choice
    for trait in sorted(trait_names):
        tiv_scores.append(df[trait])
    subjects = map(str, list(df['Subject']))
    tiv_score_dict = dict(zip(subjects, np.array(tiv_scores).transpose().tolist()))
    return tiv_score_dict, trait_names


################### util functions for processing ADACENCY MATRICES ########################

# transform an adjacency matrix with edge weights into a binary adj matrix, filtering negative weights
def get_binary_adj(graph: np.ndarray) -> np.ndarray:
    bin_adj = np.zeros(graph.shape)
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i][j] > 0.0:
                bin_adj[i][j] = 1.0
    return bin_adj


# get the bias matrices (used for MASKED ATTENTION) of all the adjacency graphs
def adj_to_bias(adj: np.ndarray, nhood: int = 1) -> np.ndarray:
    # crate an identity matrix of the same shape as the adj for current graph (it includes self-loops)
    mt = np.eye(adj.shape[1])
    # create a adj matrix to include nhood-hop neighbours
    for _ in range(nhood):
        mt = np.matmul(mt, (get_binary_adj(adj) + np.eye(adj.shape[0])))
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if mt[i][j] > 0.0:
                    mt[i][j] = 1.0
    # conjugate the adj matrix of nhood neighbours
    return -1e9 * (1.0 - mt)


# creates master node adjacency matrix attaching it as the last node
def attach_master(nb_nodes):
    mast_mat = np.zeros((nb_nodes, nb_nodes))
    for i in range(nb_nodes):
        mast_mat[nb_nodes - 1][i] = 1.0
        mast_mat[i][nb_nodes - 1] = 1.0

    return np.expand_dims(mast_mat, axis=0), np.expand_dims(adj_to_bias(mast_mat), axis=0)


# zero all the matrix entries that are not in the specified range value
def lower_bound_filter(log_10_limit: float, struct_adj: np.ndarray):
    filtered_adjs = np.empty(struct_adj.shape)
    for i in range(struct_adj.shape[0]):
        for j in range(struct_adj.shape[1]):
            filtered_adjs[i][j] = struct_adj[i][j] if ((10 ** log_10_limit) <= struct_adj[i][j]) else 0.0

    return filtered_adjs


# normalize all the adjacency matrices by row using L1 norm (unit vectors)
def norm_rows_adj(adj_mat: np.ndarray) -> np.ndarray:
    l1norm = np.abs(adj_mat.sum(axis=1))

    norm_adj = np.array((adj_mat / l1norm.reshape(adj_mat.shape[0], 1)).tolist())
    norm_adj[np.isnan(norm_adj)] = 0.0

    if not np.array_equal(get_binary_adj(adj_mat), get_binary_adj(norm_adj)):
        print("Bug in the normalization of structural adjacency matrices", file=sys.stderr)
    return norm_adj


# make an upper triangluar matrix symmetric
def make_symmetric(a: list) -> list:
    for i in range(len(a)):
        for j in range(i):
            a[i][j] = a[j][i]

    return a


# check if numpy array a is asymmetric matrix
def check_symmetric(a: np.ndarray, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


# flatten the adjacency matrices storing the edge weights and store them on disk
def persist_ew_data(get_adjs_loader):
    data_type = get_adjs_loader.__name__.split('_')[1]
    dataset_dir = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', data_type + '_data')
    ew_file = os.path.join(dataset_dir, 'flatten_edge_weigths_%s.npy' % data_type)
    if os.path.exists(ew_file):
        print('Loading the serialized edge weights data...')
        edge_weights = np.load(ew_file)
        print('Edge weights data was loaded.')
    else:
        print('Creating and serializing edge weights data...')
        adjs = list(get_adjs_loader().values())
        edge_weights = [np.array(mat)[np.triu_indices(len(mat))] for mat in adjs]
        edge_weights = np.array(edge_weights).flatten()
        np.save(ew_file, edge_weights)
        print('Edge weights data was persisted on disk.')
    return edge_weights


################### util functions for processing NODE FEATURES ########################
# a method of feature rescaling: rescale the values for each feature to (0,1)
def rescale_feats(min, max, x):
    return float(x - min) / float(max - min)


def assert_disjoint(sets):
    for set_1, set_2 in product(sets, sets):
        if set_1 is not set_2:
            assert set(set_1).isdisjoint(set(set_2))
