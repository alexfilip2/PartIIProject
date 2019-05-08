from utils.ToolsDataProcessing import *
import pickle as pkl

root_functional_data = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Data', 'functional_data')
dir_functional_data = os.path.join(root_functional_data, 'processed_data')
if not os.path.exists(dir_functional_data):
    os.makedirs(dir_functional_data)

ptnMAT_dim_sess_file = os.path.join(root_functional_data, '3T_HCP1200_MSMAll_d%d_ts2', 'netmats%d.txt')
subj_ids_file = os.path.join(root_functional_data, 'subjectIDs.txt')


def get_functional_adjacency(matrices_dim: int = 50, session_id: int = 1) -> dict:
    '''
     Retrieves the raw structural weighted adjacency matrices.
    :param matrices_dim: int specifying the dimension of the loaded matrices
    :param session_id: int the scan session from which they were generated
    :return: dict storing rank 2 ndarrays adjacency matrices, keyed by str HCP subject ID
    '''
    if matrices_dim not in [15, 25, 50, 100, 200, 300]:
        raise ValueError('Incorrect dimensionality for functional matrices: %d' % matrices_dim)
    if session_id not in [1, 2]:
        raise ValueError('Non-existent scan session for functional matrices: %d' % session_id)

    saved_processed_adjs = os.path.join(dir_functional_data,
                                        'adjacency_matrices_dim%d_sess%d.pkl' % (matrices_dim, session_id))
    if os.path.exists(saved_processed_adjs):
        print('Adjacency matrices for the functional data are already processed, loading them from disk...')
        with open(saved_processed_adjs, 'rb') as fp:
            processed_adjs = pkl.load(fp)
        print('Adjacency matrices for the functional data was loaded.')
        return processed_adjs

    print('Adjacency matrices for the functional data were not processed before, computing them now...')
    # get all the hcp subjects for which there is functional data
    subjects = []
    with open(subj_ids_file, 'r', encoding='UTF-8') as fp:
        for line in fp:
            subjects.append(line.split()[0])

    # initialize the dict as it is the first time we compute it
    processed_adjs = {}
    with open(ptnMAT_dim_sess_file % (matrices_dim, session_id), 'r', encoding='UTF-8') as fp:
        for line_nr, line in enumerate(fp):
            graph = np.zeros(shape=(matrices_dim, matrices_dim))
            for index, str_edge_weight in enumerate(line.split()):
                edge_entry = float(str_edge_weight)
                graph[index // matrices_dim][index % matrices_dim] = edge_entry
            # discard negative edge weights
            processed_adjs[subjects[line_nr]] = graph.clip(min=0.0)

    with open(saved_processed_adjs, 'wb') as fp:
        pkl.dump(processed_adjs, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Adjacency matrices for the functional data was computed and persisted on disk.')
    return processed_adjs


def get_functional_features(matrices_dim: int = 50, session_id: int = 1) -> dict:
    '''
     Retrieves the un-standardized and un-normalized node features for each functional graph.
    :param matrices_dim: int specifying the dimension of the loaded feature matrices
    :param session_id: int the scan session from which they were generated
    :return: dict storing rank 2 ndarrays node features matrices, keyed by str HCP subject ID
    '''
    from utils.Node2VecEmbedding import create_node_embedding
    if matrices_dim not in [15, 25, 50, 100, 200, 300]:
        raise ValueError('Incorrect dimensionality for functional feature matrices: %d' % matrices_dim)
    if session_id not in [1, 2]:
        raise ValueError('Non-existent scan session for functional feature matrices: %d' % session_id)
    saved_processed_feats = os.path.join(dir_functional_data,
                                         'node_features_dim%d_sess%d.pkl' % (matrices_dim, session_id))
    if os.path.exists(saved_processed_feats):
        print('Node features for the functional data are already processed, loading them from disk...')
        with open(saved_processed_feats, 'rb') as handle:
            processed_feats = pkl.load(handle)
        print('Node features for the functional data were loaded.')
        return processed_feats

    print('Creating and serializing node features for the functional data...')
    # extract the feature matrices from Node2Vec embeddings or if non-existent generate them
    node2vec_emb_dir = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Data', 'node2vec_embeds',
                                    'emb_dim%d_sess%d' % (matrices_dim, session_id))
    if not os.path.exists(node2vec_emb_dir):
        if len(os.listdir(node2vec_emb_dir)) == 0:
            create_node_embedding(matrices_dim=matrices_dim, session_id=session_id)
    # dictionary of string subject id : ndarray of shape (nr_of_nodes, nr_of_features_per_node)
    processed_feats = {}
    for embedding in os.listdir(node2vec_emb_dir):
        with open(os.path.join(node2vec_emb_dir, embedding), 'r') as fp:
            graph_format = fp.readline().split()
            nr_nodes, feat_size = int(graph_format[0]), int(graph_format[1])
            graph_feats = np.zeros((nr_nodes, feat_size))
            for _ in range(nr_nodes):
                node_features = fp.readline().split()
                node_name = int(node_features[0]) - 1
                for feat_index in range(feat_size):
                    graph_feats[node_name][feat_index] = float(node_features[feat_index])
            # retrieve the subject name and store its features matrix
            processed_feats[embedding.split('embeddings')[0]] = graph_feats

    with open(saved_processed_feats, 'wb') as fp:
        pkl.dump(processed_feats, fp, protocol=pkl.HIGHEST_PROTOCOL)
        print('Node features for the functional data were computed and persisted on disk.')
    return processed_feats


def load_funct_data(data_params: dict) -> dict:
    '''
     Retrieve the entire functional data: dict keyed by input type: adjacency matrix, attention mask, feature matrix and
      targeted scores. The example features are concatenated and ordered by their assigned HCP subject ID.
    :param data_params: dict specifying the choice of personality traits targeted, scan session and
    matrix dimensionality for functional matrices
    :return: dict containing the whole functional data-set
    '''
    # Load the dataset from disk if already formatted
    backup_file = ''.join(data_params['pers_traits_selection']).replace('NEO.NEOFAC_', '')
    backup_file = '%s_dim%d_sess%d.pkl' % (backup_file, data_params['functional_dim'], data_params['scan_session'])
    backup_file = os.path.join(dir_functional_data, backup_file)
    if os.path.exists(backup_file):
        print('Loading the serialized data set of the functional graphs...')
        with open(backup_file, 'rb') as fp:
            formatted = pkl.load(fp)
        print('Data set for the functional graphs was loaded.')
        return formatted

    # Otherwise compute the adjacency and feature matrices per HCP subject
    dict_adj = get_functional_adjacency()
    dict_node_feat = get_functional_features()
    # Format the data as a contiguous sequence of concatenated examples for each input type, also load the NEO scores
    formatted = load_data(data_params, dict_adj, dict_node_feat)

    # Save the dataset on disk
    with open(backup_file, 'wb') as fp:
        pkl.dump(formatted, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print('Data set for the functional graphs was computed and persisted on disk.')
    return formatted
