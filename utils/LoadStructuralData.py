import pickle as pkl
import operator
from utils.ToolsDataProcessing import *

root_structural_data = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Data', 'structural_data')
structural_feats_excel = os.path.join(root_structural_data, 'Features_all.xlsx')
if not os.path.exists(structural_feats_excel):
    raise IOError('Missing structural featrues file %s' % structural_feats_excel)
dir_structural_data = os.path.join(root_structural_data, 'processed_data')
if not os.path.exists(dir_structural_data):
    os.makedirs(dir_structural_data)


def get_node_names() -> dict:
    '''
     Get only the names of the nodes as brain regions which have features attached
    :return: dict representing the mapping between the str brain region and its int ID
    '''
    df = pd.read_excel(structural_feats_excel, sheet_name='nodes', index_col=None, header=None)
    brain_reg_names = list(df[1])
    brain_reg_ids = list(df[0])
    nodes_dict = dict(zip(brain_reg_names, brain_reg_ids))
    return nodes_dict


def get_feature_names() -> list:
    '''
    Get the names of node features (some of them don't appear in the dataset)
    :return: list of all feature names
    '''
    df = pd.read_excel(structural_feats_excel, sheet_name='features', index_col=None, header=None)
    features_names = sorted(list(df[0]))
    return features_names


def get_structural_adjacency() -> dict:
    '''
     Retrieves the raw structural weighted adjacency matrices.
    :return: dict storing rank 2 ndarrays adjacency matrices, keyed by str HCP subject ID
    '''

    saved_processed_adjs = os.path.join(dir_structural_data, 'adjacency_matrices.pkl')
    if os.path.exists(saved_processed_adjs):
        print('Adjacency matrices for the structural data are already processed, loading them from disk...')
        with open(saved_processed_adjs, 'rb') as fp:
            processed_adjs = pkl.load(fp)
        print('Adjacency matrices for the structural data were loaded.')
        return processed_adjs

    print('Adjacency matrices for the structural data were not processed before, computing them now...')
    dir_structural_adjs = os.path.join(root_structural_data, 'PTN_matrices')
    # os.walk includes as the first item the parent directory itself then the rest of sub-directories
    if not os.listdir(dir_structural_adjs):
        raise IOError('Raw structural adjacency data is missing from directory %s' % dir_structural_adjs)
    subjects_subdirs = [os.path.join(dir_structural_adjs, subdir) for subdir in next(os.walk(dir_structural_adjs))[1]]
    # the brain region ID's of all nodes that have node features
    nodes_with_data = set(get_node_names().values())

    # initialize the dict as it is the first time we compute it
    processed_adjs = {}
    for subject_dir in subjects_subdirs:
        if not os.listdir(subject_dir):
            raise IOError('Adjacency matrix for subject % is missing' % subject_dir)
        for subj_id in os.listdir(subject_dir):
            with open(os.path.join(subject_dir, subj_id), 'r', encoding='UTF-8') as fp:
                graph = []
                for row_index, line in enumerate(fp, start=1):
                    if row_index not in nodes_with_data:
                        continue
                    graph_row = []
                    for col_index, edge_weight in enumerate(line.split(), start=1):
                        if col_index not in nodes_with_data:
                            continue
                        graph_row.append(float(edge_weight))
                    graph.append(graph_row)
            hcp_subject_id = subj_id.split('_')[0]
            processed_adjs[hcp_subject_id] = make_symmetric(np.array(graph))

    # persist the adjacency matrices dict on disk for further use
    with open(saved_processed_adjs, 'wb') as fp:
        pkl.dump(processed_adjs, fp, protocol=pkl.HIGHEST_PROTOCOL)
        print('Adjacency matrices for the structural data were computed and persisted on disk.')
    return processed_adjs


def get_structural_features() -> dict:
    '''
      Retrieves the un-standardized and un-normalized node features for each structural graph.
    :return: dict storing rank 2 ndarrays node features matrices, keyed by str HCP subject ID
    '''
    saved_processed_feats = os.path.join(dir_structural_data, 'node_features.pkl')
    if os.path.exists(saved_processed_feats):
        print('Node features for the structural data are already processed, loading them from disk...')
        with open(saved_processed_feats, 'rb') as fp:
            processed_feats = pkl.load(fp)
        print('Node features for the structural data were loaded.')
        return processed_feats

    print('Node features for the structural data were not processed before, computing them now...\n')
    # DataFrame object containing the data of the 'Data' sheet in the Excel dataset
    df = pd.read_excel(structural_feats_excel, sheet_name='Data')
    # dictionary of string subject id : ndarray of shape (nr_of_nodes, nr_of_features_per_node)
    processed_feats = {}
    # all feature names (for some of them we DO NOT HAVE data)
    all_feats_names = get_feature_names()
    # list of (node name, subject ID) sorted by ID
    brain_regs = sorted(get_node_names().items(), key=operator.itemgetter(1))
    # the names of features for which THERE IS DATA per each node
    present_feats = set([])
    # find the range of values for each feature and for which of these there is data on each node
    for row_index, graph_data in df.iterrows():
        subj_graph_feats = []
        for n_name, n_id in brain_regs:
            current_node_feat = []
            for f_name in all_feats_names:
                # check if there is data for the (node, feature) in the header of the Excel sheet
                feat_region_name = 'fs%s_%s' % (n_name, f_name)
                if feat_region_name in graph_data.keys():
                    present_feats.add(f_name)
                    current_node_feat.append(float(graph_data[feat_region_name]))
            subj_graph_feats.append(current_node_feat)
        processed_feats[str(int(graph_data['Subjects']))] = np.array(subj_graph_feats)

    with open(saved_processed_feats, 'wb') as fp:
        pkl.dump(processed_feats, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Node features for the structural data were computed and persisted on disk.')

    return processed_feats


def load_struct_data(data_params: dict) -> dict:
    '''
     Retrieve the entire structural data: dict keyed by str HCP subject ID storing a dict with each specific input for
    the subject; adjacency matrix, attention mask, feature matrix, targeted scores
    :param data_params: dict specifying the choice of personality traits targeted, threshold filter and its lower bound
     for the edge weights
    :return: dict containing the whole data-set
    '''
    str_traits = ''.join([trait.replace('NEO.NEOFAC_', '') for trait in data_params['pers_traits_selection']])
    str_limits = '' if data_params['edgeWeights_filter'] is None else str(data_params['low_ew_limit'])
    saved_data_file = os.path.join(dir_structural_data, '%s_%s.pkl' % (str_traits, str_limits))
    if os.path.exists(saved_data_file):
        print('Loading the serialized data set of structural graphs...')
        with open(saved_data_file, 'rb') as fp:
            data = pkl.load(fp)
        print('Data set for the structural graphs was loaded.')
        return data

    dict_adj = get_structural_adjacency()
    dict_node_feat = get_structural_features()
    dict_tiv_score = get_NEO5_scores(data_params['pers_traits_selection'])
    dict_data = {}
    all_subjects = sorted(list(dict_adj.keys()))
    for subj_id in all_subjects:
        if subj_id in dict_node_feat.keys() and subj_id in dict_tiv_score.keys():
            dict_data[subj_id] = {}
            dict_data[subj_id]['bias_in'] = adj_to_bias(dict_adj[subj_id], nhood=1)
            # normalize the rows of the adjacency matrix, apply threshold filter for the edge weights
            if data_params['edgeWeights_filter'] is None:
                norm_adj = norm_rows_adj(dict_adj[subj_id])
            else:
                norm_adj = norm_rows_adj(
                    data_params['edgeWeights_filter'](dict_adj[subj_id], data_params['low_ew_limit']))
            dict_data[subj_id]['adj_in'] = norm_adj
            dict_data[subj_id]['ftr_in'] = dict_node_feat[subj_id]
            dict_data[subj_id]['score_in'] = dict_tiv_score[subj_id]
    # standardise and normalize the raw node features
    preprocess_features(dict_data)

    with open(saved_data_file, 'wb') as fp:
        pkl.dump(dict_data, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Data set for the structural graphs was computed and persisted on disk.')

    return dict_data
