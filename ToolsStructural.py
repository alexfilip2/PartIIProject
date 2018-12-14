from Tools import *

np.set_printoptions(threshold=np.nan)
from sys import stderr

dir_root_structural_data = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'structural_data')
dir_struct_mat_HCP = os.path.join(dir_root_structural_data, 'PTN matrices HCP')
structural_feats_excel = os.path.join(dir_root_structural_data, 'Features_all.xlsx')


# dictionary of node names-brain regions (with features attached) of structural graphs and their ID's among all nodes
def get_struct_n_names():
    df = pd.read_excel(structural_feats_excel, sheet_name='nodes', index_col=None, header=None)
    brain_reg_names = list(df[1])
    brain_reg_ids = list(df[0])
    nodes_dict = dict(zip(brain_reg_names, brain_reg_ids))
    return nodes_dict


# list of string names of the node features (some of them are not present in the dataset)
def get_struct_f_names():
    df = pd.read_excel(structural_feats_excel, sheet_name='features', index_col=None, header=None)
    features_names = sorted(list(df[0]))
    return features_names


# a method of feature rescaling: rescale the values for each feature to (0,1)
def rescale_feats(min, max, x):
    return float(x - min) / float(max - min)


def get_struct_node_feat():
    # DataFrame object containing the data of the 'Data' sheet in the Excel dataset
    df = pd.read_excel(structural_feats_excel, sheet_name='Data')
    # dictionary of string patient id : array of shape (nr_of_nodes, nr_of_features_per_node)
    all_node_feats = {}
    # all feature names (for some of them we DO NOT HAVE data)
    feat_names = get_struct_f_names()
    # list of (node name, subject ID) sorted by ID
    brain_regs = sorted(get_struct_n_names().items(), key=operator.itemgetter(1))
    # the names of features for which THERE IS DATA per each node
    present_feats = set([])
    # the range of scalar values for each node feature
    f_value_range = {}
    # find the range of values for each feature and for which of these there is data on each node
    for row_index, graph_data in df.iterrows():
        for n_name, n_id in brain_regs:
            for f_name in feat_names:
                # check if there is data for the (node, feature) in the header of the Excel sheet
                feat_brainreg_name = 'fs' + n_name + '_' + f_name
                if feat_brainreg_name in graph_data.keys():
                    present_feats.add(f_name)
                    if f_name not in f_value_range.keys():
                        f_value_range[f_name] = [float(graph_data[feat_brainreg_name])]
                    else:
                        f_value_range[f_name].append(float(graph_data[feat_brainreg_name]))

    present_feats = sorted(list(present_feats))
    # min/max scalar values for each feature that we have data for
    f_value_limits = {}
    for feat_name in present_feats:
        if feat_name not in f_value_limits.keys():
            f_value_limits[feat_name] = {}
        f_value_limits[feat_name]['min'] = min(f_value_range[feat_name])
        f_value_limits[feat_name]['max'] = max(f_value_range[feat_name])

    # extracting for each subject the feature vector of each node
    for row_index, graph_data in df.iterrows():
        current_graph_feats = []
        for n_name, n_id in brain_regs:
            curr_node_feat = []
            for f_name in present_feats:
                feat_brainreg_name = 'fs' + n_name + '_' + f_name
                curr_node_feat.append(rescale_feats(f_value_limits[f_name]['min'],
                                                    f_value_limits[f_name]['max'],
                                                    float(graph_data[feat_brainreg_name])))
            current_graph_feats.append(curr_node_feat)

        all_node_feats[str(int(graph_data['Subjects']))] = current_graph_feats

    return all_node_feats


# check if numpy array a is asymmetric matrix
def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def get_filtered_struct_adjs(limit=sys.float_info.max):
    # os.walk includes as the first item the parent directory itself then the rest of sub-directories
    subjects_subdirs = [os.path.join(dir_struct_mat_HCP, subdir) for subdir in next(os.walk(dir_struct_mat_HCP))[1]]
    # the brain region ID's of all nodes that have node features
    filtered_nodes = get_struct_n_names().values()

    adj = {}
    for subject_dir in subjects_subdirs:
        for subj_id in os.listdir(subject_dir):
            with open(os.path.join(subject_dir, subj_id), 'r', encoding='UTF-8') as subj_data:
                graph = []
                for row_index, line in enumerate(subj_data, start=1):
                    if row_index not in filtered_nodes: continue
                    adj_row = []
                    for col_index, edge_weight in enumerate(line.split(), start=1):
                        if col_index not in filtered_nodes: continue
                        adj_row.append(
                            float(edge_weight) if float(edge_weight) < limit else 0.0)
                    graph.append(adj_row)
            # the adjancency matrices are upper diagonal, we make them symmetric
            i_lower = np.tril_indices(len(graph), -1)
            sym_adj = np.array(graph)
            sym_adj[i_lower] = sym_adj.T[i_lower]
            if not check_symmetric(sym_adj): print("Making this adjancency matrix symmetric failed", file=stderr)
            # normalize the rows of the adjacency matrix

            adj[subj_id.split('_')[0]] = sym_adj.tolist()

    return adj


def load_struct_data(trait_choice=None, ew_limit=500000):
    dict_adj = get_filtered_struct_adjs(ew_limit)
    dict_node_feat = get_struct_node_feat()
    dict_tiv_score = get_NEO5_scores(trait_choice)

    adj_matrices, graph_features, scores = [], [], []
    subjects = sorted(list(dict_adj.keys()))
    for subj_id in subjects:
        if subj_id in dict_node_feat.keys() and subj_id in dict_tiv_score.keys():
            adj_matrices.append(dict_adj[subj_id])
            graph_features.append(dict_node_feat[subj_id])
            scores.append(dict_tiv_score[subj_id])

    data_sz = len(scores)

    score_train, score_val, score_test = np.split(np.array(scores), [int(data_sz * 0.8), int(data_sz * 0.9)])
    return np.array(adj_matrices), np.array(graph_features), score_train, score_val, score_test


def mat_flatten(mat):
    return np.squeeze(np.matrix(mat).flatten())


def load_regress_data(trait_choice):
    dict_adj = get_filtered_struct_adjs()
    dict_tiv_score = get_NEO5_scores(trait_choice)
    adj_matrices, scores = [], []
    for subj_id in sorted(list(dict_adj.keys())):
        if subj_id in dict_tiv_score.keys():
            adj_matrices.append(mat_flatten(dict_adj[subj_id]).tolist()[0])
            scores.append(dict_tiv_score[subj_id])

    return np.array(adj_matrices), np.array(scores)


if __name__ == "__main__":
    adjs = get_filtered_struct_adjs()
    print(np.abs(np.array(adjs['100206'])).sum(axis=1))
