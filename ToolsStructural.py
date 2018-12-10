import numpy as np
import operator

np.set_printoptions(threshold=np.nan)
import os
import pandas as pd
import random

struct_mat_root = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'structural_mat')
struct_mat_HCP = os.path.join(struct_mat_root, 'PTN matrices HCP')
struct_feat = os.path.join(struct_mat_root, 'Features_all.xlsx')

pers_scores = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'TIVscores',
                           '1016_HCP_withTIV_acorrected_USETHIS.xlsx')


def rescale_feats(min, max, x):
    return float(x - min) / float(max - min)


def get_struct_nodes():
    df = pd.read_excel(struct_feat, sheet_name='nodes', index_col=None, header=None)

    brain_reg_names = list(df[1])
    brain_reg_ids = list(df[0])
    nodes_dict = dict(zip(brain_reg_names, brain_reg_ids))
    return nodes_dict


def get_feat_names():
    df = pd.read_excel(struct_feat, sheet_name='features', index_col=None, header=None)
    features_names = sorted(list(df[0]))
    return features_names


def get_graphs_feat():
    # DataFrame object containing the data of the 'Data' sheet in the Excel dataset
    df = pd.read_excel(struct_feat, sheet_name='Data')
    # dictionary of string patient id : array of shape nr_of_nodes X  nr_of_features_per_node
    graph_feats = {}
    # all feature names (for some of them we DO NOT HAVE data)
    all_feats = get_feat_names()
    # node names - brain regions (just a part of them) sorted by their ID's into the set of all brain regions
    nodes = sorted(get_struct_nodes().items(), key=operator.itemgetter(1))
    # the names of features for which WE HAVE DATA
    unique_feats = set([])
    # the range of scalar values for each node feature
    feat_values = {}
    for row_index, row in df.iterrows():
        for n_name, n_id in nodes:
            for f_name in all_feats:
                # check if there is data for the node, feature pair in the header of the Excel sheet
                if ('fs' + n_name + '_' + f_name) in row.keys():
                    unique_feats.add(f_name)
                    if f_name not in feat_values.keys():
                        feat_values[f_name] = [float(row['fs' + n_name + '_' + f_name])]
                    else:
                        feat_values[f_name].append(float(row['fs' + n_name + '_' + f_name]))

    unique_feats = sorted(list(unique_feats))
    # min/max scalar values for each feature that we have data for
    feat_limits = {}
    for feat_name in unique_feats:
        if feat_name not in feat_limits.keys():
            feat_limits[feat_name] = {}
        feat_limits[feat_name]['min'] = min(feat_values[feat_name])
        feat_limits[feat_name]['max'] = max(feat_values[feat_name])

    for row_index, row in df.iterrows():
        node_feat_vecs = []
        for n_name, n_id in nodes:
            curr_node_feat = []
            for f_name in unique_feats:
                curr_node_feat.append(rescale_feats(feat_limits[f_name]['min'],
                                                    feat_limits[f_name]['max'],
                                                    float(row['fs' + n_name + '_' + f_name])))
            node_feat_vecs.append(curr_node_feat)
        graph_feats[str(int(row['Subjects']))] = node_feat_vecs

    return graph_feats


def get_filter_strcut_adjs():
    all_sub_dirs = [os.path.join(struct_mat_HCP, subdir) for subdir in next(os.walk(struct_mat_HCP))[1]]
    filtered_nodes = get_struct_nodes().values()

    adj = {}
    for subdir in all_sub_dirs:
        for subj_id in os.listdir(subdir):
            with open(os.path.join(subdir, subj_id), 'r', encoding='UTF-8') as data:
                graph = []
                for row, line in enumerate(data):
                    if row not in filtered_nodes: continue
                    adj_row = []
                    for column, edge_weight in enumerate(line.split()):
                        if column not in filtered_nodes: continue
                        adj_row.append(float(edge_weight))
                    graph.append(adj_row)
            i_lower = np.tril_indices(len(graph), -1)
            matrix = np.array(graph)
            matrix[i_lower] = matrix.T[i_lower]
            adj[subj_id.split('_')[0]] = matrix.tolist()

    return adj


all_pers_traits = ['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E']


def get_NEO5_scores(trait_names=all_pers_traits):
    df = pd.ExcelFile(pers_scores).parse('Raw_data')  # you could add index_col=0 if there's an index
    tiv_scores = []
    for trait in trait_names:
        tiv_scores.append(df[trait])
    subjects = map(str, list(df['Subject']))
    tiv_score_dict = dict(zip(subjects, np.array(tiv_scores).transpose().tolist()))

    return tiv_score_dict


def load_struct_data(trait_choice):
    dict_adj = get_filter_strcut_adjs()
    dict_node_feat = get_graphs_feat()
    dict_tiv_score = get_NEO5_scores(trait_choice)
    adj_matrices, graph_features, scores = [], [], []
    for subj_id in sorted(list(dict_adj.keys())):
        if subj_id in dict_node_feat.keys() and subj_id in dict_tiv_score.keys():
            adj_matrices.append(dict_adj[subj_id])
            graph_features.append(dict_node_feat[subj_id])
            scores.append(dict_tiv_score[subj_id])

    data_scores = np.array(scores)
    score_train, score_val, score_test = np.split(data_scores,
                                                  [int(len(data_scores) * 0.8), int(len(data_scores) * 0.9)])
    return np.array(adj_matrices), np.array(graph_features), score_train, score_val, score_test

def load_regress_data(trait_choice):
    dict_adj = get_filter_strcut_adjs()
    dict_tiv_score = get_NEO5_scores(trait_choice)
    adj_matrices, scores = [], []
    for subj_id in sorted(list(dict_adj.keys())):
        if subj_id in dict_tiv_score.keys():
            adj_matrices.append(mat_flatten(dict_adj[subj_id]).tolist()[0])
            scores.append(dict_tiv_score[subj_id])

    return np.array(adj_matrices), np.array(scores)

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def mat_flatten(mat):
    return  np.squeeze(np.matrix(mat).flatten())

if __name__ == "__main__":
    adjs = get_filter_strcut_adjs()['100206']
    print(len(adjs))
    print(adjs.shape)

