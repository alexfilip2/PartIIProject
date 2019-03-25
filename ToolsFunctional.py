from Tools import *
from Node2Vec import *

ptnMAT_colab = join(os.getcwd(), os.pardir, 'PartIIProject', 'functional_data')
dir_proc_funct_data = join(ptnMAT_colab, 'processed_data')
if not os.path.exists(dir_proc_funct_data):
    os.makedirs(dir_proc_funct_data)

ptnMAT_dim_sess_file = join(ptnMAT_colab, '3T_HCP1200_MSMAll_d%d_ts2', 'netmats%d.txt')
subj_id_file = join(ptnMAT_colab, 'subjectIDs.txt')


def get_functional_adjs(matrices_dim=50, session_id=1):
    adjs_file = os.path.join(dir_proc_funct_data, 'adjs_matrices_dim%d_sess%d.pkl' % (matrices_dim, session_id))
    if os.path.exists(adjs_file):
        print('Loading the serialized adjacency matrices for the functional data...')
        with open(adjs_file, 'rb') as handle:
            dict_adj = pkl.load(handle)
        print('Adjacency matrices for the functional data was loaded.')
        print(dict_adj)
        return dict_adj

    print('Creating and serializing adjacency matrices for functional data...')
    subj_ids = []
    with open(subj_id_file, 'r', encoding='UTF-8') as data:
        for line in data:
            subj_ids.append(line.split()[0])
    dict_adj = {}
    with open(ptnMAT_dim_sess_file % (matrices_dim, session_id), 'r', encoding='UTF-8') as data:
        for line_nr, line in enumerate(data):
            graph = np.zeros(shape=(matrices_dim, matrices_dim))
            for index, str_edge_weight in enumerate(line.split()):
                edge_entry = float(str_edge_weight)
                graph[index // matrices_dim][index % matrices_dim] = edge_entry
            # discard negative edge weights
            dict_adj[subj_ids[line_nr]] = graph.clip(min=0.0)

    with open(adjs_file, 'wb') as handle:
        pkl.dump(dict_adj, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print('Adjacency matrices for the functional data was computed and persisted on disk.')

    return dict_adj


def get_functional_node_feat(matrices_dim=50, session_id=1):
    node_feats_file = os.path.join(dir_proc_funct_data, 'node_feats_dim%d_sess%d.pkl' % (matrices_dim, session_id))
    if os.path.exists(node_feats_file):
        print('Node features for the functional data already processed, loading them from disk...')
        with open(node_feats_file, 'rb') as handle:
            all_node_feats = pkl.load(handle)
        print('Node features for the functional data was loaded.')
        return all_node_feats

    print('Creating and serializing node features for the functional data...')
    node2vec_emb_dir = join(os.getcwd(), os.pardir, 'PartIIProject', 'node2vec_embeds',
                            'emb_dim%d_sess%d' % (matrices_dim, session_id))
    if not os.path.exists(node2vec_emb_dir):
        create_node_embedding(matrices_dim=matrices_dim, session_id=session_id)
    all_node_feats = {}
    feats_limits = {}
    feat_size = 0
    for embed in os.listdir(node2vec_emb_dir):
        with open(join(node2vec_emb_dir, embed), 'r') as emb_handle:
            format = emb_handle.readline().split()
            nr_nodes, feat_size = int(format[0]), int(format[1])
            graph_feats = np.zeros((nr_nodes, feat_size))
            for _ in range(nr_nodes):
                node_str_feat = emb_handle.readline().split()
                curr_node = int(node_str_feat[0]) - 1
                for feat_index in range(stop=feat_size + 1, start=1):
                    curr_feat_val = float(node_str_feat[feat_index])
                    graph_feats[curr_node][feat_index] = curr_feat_val
                    if feat_index not in feats_limits.keys():
                        feats_limits[feat_index]['min'] = curr_feat_val
                        feats_limits[feat_index]['max'] = curr_feat_val
                    else:
                        feats_limits[feat_index]['min'] = min(curr_feat_val, feats_limits[feat_index]['min'])
                        feats_limits[feat_index]['max'] = max(curr_feat_val, feats_limits[feat_index]['max'])
            # retrieve the subject name and stroe its features matrix
            all_node_feats[embed.split('embeddings')[0]] = graph_feats

    for subj in all_node_feats.keys():
        for node_vect in all_node_feats[subj]:
            for feat_index in range(feat_size):
                node_vect[feat_index] = rescale_feats(feats_limits[feat_index]['min'],
                                                      feats_limits[feat_index]['max'],
                                                      node_vect[feat_index])
    with open(node_feats_file, 'wb') as handle:
        pkl.dump(all_node_feats, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print('Node features for the functional data was computed and persisted on disk.')

    return all_node_feats


def load_funct_data(hyparams):
    str_traits = ''.join([trait.split('NEO.NEOFAC_')[-1] for trait in hyparams['pers_traits_selection']])
    binary_prefix = '%s_d%d_s%d.pkl' % (str_traits, hyparams['functional_dim'], hyparams['scan_session'])

    dataset_binary = join(dir_proc_funct_data, binary_prefix)
    if os.path.exists(dataset_binary):
        print('Loading the serialized data for the functional graphs...')
        with open(dataset_binary, 'rb') as handle:
            data = pkl.load(handle)
        print('Data set for the functional graphs was loaded.')
        return data['data'], data['subjs']

    dict_adj = get_functional_adjs()
    dict_node_feat = get_functional_node_feat()
    dict_tiv_score, _ = get_NEO5_scores(hyparams['pers_traits_selection'])

    dict_dataset = {}
    available_subjs = []
    subjects = sorted(list(dict_adj.keys()))
    for subj_id in subjects:
        if subj_id in dict_node_feat.keys() and subj_id in dict_tiv_score.keys():
            dict_dataset[subj_id] = {}
            dict_dataset[subj_id]['adj'] = norm_rows_adj(dict_adj[subj_id])
            dict_dataset[subj_id]['feat'] = dict_node_feat[subj_id]
            dict_dataset[subj_id]['bias'] = adj_to_bias(dict_adj[subj_id], nhood=1)
            dict_dataset[subj_id]['score'] = dict_tiv_score[subj_id]
            available_subjs.append(subj_id)

    with open(dataset_binary, 'wb') as handle:
        pkl.dump({'data': dict_dataset, 'subjs': sorted(available_subjs)}, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print('Data set for the functional graphs was computed and persisted on disk.')

    return dict_dataset, available_subjs
