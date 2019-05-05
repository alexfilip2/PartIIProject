from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import sys


def get_NEO5_scores(trait_choice: list) -> dict:
    '''
    Retrieves the personality scores data from disk.
    :param trait_choice: list of names of personality traits
    :return: dict storing ndarray of #traits float values, keyed by str HCP subject ID
    '''
    # Excel file storing personality scores for each HCP subject
    pers_scores_file = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Data', 'TIVscores',
                                    '1016_HCP_TIVscores.xlsx')

    if os.path.exists(pers_scores_file):
        df = pd.ExcelFile(pers_scores_file).parse('Raw_data')
        tiv_scores = []
        # retrieve each column data for individual traits
        for trait in trait_choice:
            tiv_scores.append(df[trait])
        # build the dict
        subjects = list(map(str, list(df['Subject'])))
        tiv_score_dict = dict(zip(subjects, np.array(tiv_scores).transpose()))
        for i in range(len(tiv_scores)):
            for j in range(len(tiv_scores[i])):
                assert tiv_score_dict[subjects[j]][i] == tiv_scores[i][j]
    else:
        raise IOError('Missing personality scores Excel file %s' % pers_scores_file)

    return tiv_score_dict


def get_binary_adj(adj: np.ndarray) -> np.ndarray:
    '''
    Generates the binary adjacency from a weighted one - only the positive entries represent connections
    :param adj: rank 2 ndarray representing a weighted adjacency matrix
    :return: its binary adjacency matrix as a rank 2 ndarray
    '''
    assert len(adj.shape) == 2
    bin_adj = np.zeros(adj.shape)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] > 0.0:
                bin_adj[i][j] = 1.0
    return bin_adj


def adj_to_bias(adj: np.ndarray, nhood: int = 1) -> np.ndarray:
    '''
    Generates the mask adjacency from the weighted one - used by masked attention in the GAT layer
    :param adj: rank 2 ndarray representing a weighted adjacency matrix
    :param nhood: maximum number of hops from a target node where nodes are still considered its neighbours
    :return: rank 2 ndarray representing the mask adjacency
    '''
    assert len(adj.shape) == 2
    # crate an identity matrix of the same shape as the adj for current graph (it includes self-loops)
    neighbourhood = np.eye(adj.shape[1])
    # create a adj matrix to include nhood-hop neighbours
    for _ in range(nhood):
        neighbourhood = np.matmul(neighbourhood, (get_binary_adj(adj) + np.eye(adj.shape[0])))
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if neighbourhood[i][j] > 0.0:
                    neighbourhood[i][j] = 1.0
    # conjugate the adj matrix of nhood neighbours
    return -1e9 * (1.0 - neighbourhood)


def lower_bound_filter(struct_adj: np.ndarray, log_10_limit: float = 2.4148) -> np.ndarray:
    '''
    Reduce to 0.0 all the entries of a structural adjacency matrix that don't reach a specific threshold
    :param log_10_limit: logarithm base 10 of the threshold value
    :param struct_adj: rank 2 ndarray representing a weighted adjacency matrix of a structural graph
    :return: rank 2 ndarray representing the filtered weighted adjacency matrix
    '''
    filtered_adj = np.empty(struct_adj.shape)
    for i in range(struct_adj.shape[0]):
        for j in range(struct_adj.shape[1]):
            filtered_adj[i][j] = struct_adj[i][j] if ((10 ** log_10_limit) <= struct_adj[i][j]) else 0.0
    return filtered_adj


def norm_rows_adj(adj_mat: np.ndarray) -> np.ndarray:
    '''
     Normalize the last axis of a weighted adjacency matrix by row using L1 norm
    :param adj_mat:  rank 2 ndarray representing a weighted adjacency matrix
    :return:  rank 2 ndarray representing the normalized adjacency matrix
    '''
    l1norm = np.abs(adj_mat.sum(axis=1))
    norm_adj = np.array((adj_mat / l1norm.reshape(adj_mat.shape[0], 1)).tolist())
    # in case of disconnected nodes, preserve the 0.0 entries
    norm_adj[np.isnan(norm_adj)] = 0.0
    if not np.array_equal(get_binary_adj(adj_mat), get_binary_adj(norm_adj)):
        print("The normalization of the current adjacency matrix failed.", file=sys.stderr)
    return norm_adj


def make_symmetric(upper_triang_adj: np.ndarray) -> np.ndarray:
    '''
     Make an upper triangular matrix symmetric (used for the functional graphs)
    :param upper_triang_adj: upper triangular matrix
    :return: symmetric adjacency matrix
    '''
    for i in range(len(upper_triang_adj)):
        for j in range(i):
            upper_triang_adj[i][j] = upper_triang_adj[j][i]

    def check_symmetric(a: np.ndarray, tol=1e-8):
        '''
        Checks if ndarray is a symmetric matrix
        :param a: rank 2 ndarray
        :param tol: threshold of error
        :return: bool result of the checking
        '''
        return np.allclose(a, a.T, atol=tol)

    assert check_symmetric(upper_triang_adj)
    return upper_triang_adj


def preprocess_features(entire_data: dict):
    '''
    Standardise the node features and then normalize them
    :param entire_data: dict of the entire data of adjacency matrices, masks etc indexed by HCP subject ID's
    :return: void as the dict object is mutable and updated directly
    '''
    # keep a fixed order between the example graphs
    subjects = sorted(list(entire_data.keys()))
    data_sz = len(subjects)
    N, F = entire_data[subjects[0]]['ftr_in'].shape
    # unpack the feature matrices into individual node features to standardise a node feature, not example graph
    concat_features = np.concatenate([entire_data[subject]['ftr_in'] for subject in subjects])
    standardised_feats = MinMaxScaler().fit_transform(X=concat_features).reshape((data_sz, N, F))
    for example_index, subject in enumerate(subjects):
        entire_data[subject]['ftr_in'] = standardised_feats[example_index]
