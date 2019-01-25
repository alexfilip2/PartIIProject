from Tools import *
import pickle as pkl

np.set_printoptions(threshold=np.nan)

dir_root_structural_data = join(os.getcwd(), os.pardir, 'PartIIProject', 'structural_data')
dir_struct_mat_HCP = join(dir_root_structural_data, 'PTN matrices HCP')
structural_feats_excel = join(dir_root_structural_data, 'Features_all.xlsx')
dir_proc_struct_data = join(dir_root_structural_data, 'processed_data')
if not os.path.exists(dir_proc_struct_data):
    os.makedirs(dir_proc_struct_data)


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


def get_scaled_struct_node_feat():
    node_feats_binary = join(dir_proc_struct_data, 'node_feats.pkl')
    if os.path.exists(node_feats_binary):
        print('Node features for the structural data already processed, loading them from disk...')
        with open(node_feats_binary, 'rb') as handle:
            all_node_feats = pkl.load(handle)
        print('Node features for the structural data was loaded.')
        return all_node_feats

    print('Creating and serializing node features data for the structural data...')
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

    with open(node_feats_binary, 'wb') as handle:
        pkl.dump(all_node_feats, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print('Node features for the structural data was computed and persisted on disk.')

    return all_node_feats


def get_structural_adjs():
    adjs_binary = join(dir_proc_struct_data, 'adjs_matrices.pkl')
    if os.path.exists(adjs_binary):
        print('Loading the serialized adjacency matrices for the structural data...')
        with open(adjs_binary, 'rb') as handle:
            all_adjs = pkl.load(handle)
        print('Adjacency matrices for the structural data was loaded.')
        return all_adjs

    print('Creating and serializing adjacency matrices for structural data...')
    # os.walk includes as the first item the parent directory itself then the rest of sub-directories
    subjects_subdirs = [join(dir_struct_mat_HCP, subdir) for subdir in next(os.walk(dir_struct_mat_HCP))[1]]
    # the brain region ID's of all nodes that have node features
    filtered_nodes = get_struct_n_names().values()

    all_adjs = {}
    for subject_dir in subjects_subdirs:
        for subj_id in os.listdir(subject_dir):
            with open(join(subject_dir, subj_id), 'r', encoding='UTF-8') as subj_data:
                graph = []
                for row_index, line in enumerate(subj_data, start=1):
                    if row_index not in filtered_nodes: continue
                    adj_row = []
                    for col_index, edge_weight in enumerate(line.split(), start=1):
                        if col_index not in filtered_nodes: continue
                        adj_row.append(float(edge_weight))
                    graph.append(adj_row)
            # the adjancency matrices are upper diagonal, we make them symmetric
            all_adjs[subj_id.split('_')[0]] = make_symmetric(graph)

    with open(adjs_binary, 'wb') as handle:
        pkl.dump(all_adjs, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print('Adjacency matrices for the structural data was computed and persisted on disk.')
    return all_adjs


def load_struct_data(model_GAT_choice=None):
    dataset_binary = join(dir_proc_struct_data, 'dataset.pkl')
    if os.path.exists(dataset_binary):
        print('Loading the serialized data for the structural graphs...')
        with open(dataset_binary, 'rb') as handle:
            data = pkl.load(handle)
        print('Data set for the structural graphs was loaded.')
        return data['data'], data['subjs']

    dict_adj = get_structural_adjs()
    dict_node_feat = get_scaled_struct_node_feat()
    dict_tiv_score = get_NEO5_scores(model_GAT_choice.pers_traits)

    dict_dataset = {}
    all_subjects = sorted(list(dict_adj.keys()))
    available_subjs = []
    for subj_id in all_subjects:
        if subj_id in dict_node_feat.keys() and subj_id in dict_tiv_score.keys():
            dict_dataset[subj_id] = {}
            unexp_adj = norm_rows_adj(model_GAT_choice.filter(model_GAT_choice.limits, np.array(dict_adj[subj_id])))
            if np.isnan(unexp_adj).any():
                print(unexp_adj)
                quit()
            dict_dataset[subj_id]['feat'] = exp_dims(np.array(dict_node_feat[subj_id]), axis=0)
            dict_dataset[subj_id]['adj'] = exp_dims(unexp_adj, axis=0)
            dict_dataset[subj_id]['bias'] = exp_dims(adj_to_bias(unexp_adj, nhood=1), axis=0)
            dict_dataset[subj_id]['score'] = exp_dims(np.array(dict_tiv_score[subj_id]), axis=0)
            available_subjs.append(subj_id)

    with open(dataset_binary, 'wb') as handle:
        pkl.dump({'data': dict_dataset, 'subjs': sorted(available_subjs)}, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print('Data set for the structural graphs was computed and persisted on disk.')

    return dict_dataset, sorted(available_subjs)


def load_regress_data(trait_choice):
    data, subjs = load_struct_data()
    adj_matrices, scores = [], []
    traits = np.array(sorted(['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E']))
    (trait_index,) = np.where(traits == trait_choice)
    for subj_id in sorted(subjs):
        adj_matrices.append(mat_flatten(data[subj_id]['adj']).tolist())
        scores.append(data[subj_id]['score'][0][trait_index])

    return np.squeeze(np.array(adj_matrices), axis=1), np.squeeze(np.array(scores), axis=1)
