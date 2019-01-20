from Tools import *

PTN_MAT_DIM = 50
ptnMAT_colab = join(os.getcwd(), os.pardir, 'PartIIProject', 'functional_data')
ptnMAT_d50_dir = join(ptnMAT_colab, '3T_HCP1200_MSMAll_d50_ts2')

ptnMAT_d50_ses1 = join(ptnMAT_d50_dir, 'netmats1.txt')
ptnMAT_d50_ses2 = join(ptnMAT_d50_dir, 'netmats2.txt')
subj_id_file = join(ptnMAT_colab, 'subjectIDs.txt')


def get_functional_adjs(ptn_dim=50, sess_file=ptnMAT_d50_ses1):
    adjs_file = os.path.join(ptnMAT_colab, 'adjs_matrices.pkl')
    if os.path.exists(adjs_file):
        print('Loading the serialized adjacency matrices for the functional data...')
        with open(adjs_file, 'rb') as handle:
            dict_adj = pkl.load(handle)
        print('Adjacency matrices for the functional data was loaded.')
        return dict_adj

    print('Creating and serializing adjacency matrices for functional data...')
    subj_ids = []
    with open(subj_id_file, 'r', encoding='UTF-8') as data:
        for line in data:
            subj_ids.append(line.split()[0])
    dict_adj = {}
    with open(sess_file, 'r', encoding='UTF-8') as data:
        for line_nr, line in enumerate(data):
            graph = [[0 for x in range(ptn_dim)] for y in range(ptn_dim)]
            for index, edge_weight in enumerate(line.split()):
                graph[int(index / ptn_dim)][int(index % ptn_dim)] = float(edge_weight) if float(
                    edge_weight) > 0 else 0.0
            dict_adj[subj_ids[line_nr]] = np.array(graph)

    with open(adjs_file, 'wb') as handle:
        pkl.dump(dict_adj, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print('Adjacency matrices for the functional data was computed and persisted on disk.')

    return dict_adj


def get_functional_node_feat():
    node_feats_file = os.path.join(ptnMAT_colab, 'node_feats.pkl')
    if os.path.exists(node_feats_file):
        print('Node features for the functional data already processed, loading them from disk...')
        with open(node_feats_file, 'rb') as handle:
            all_node_feats = pkl.load(handle)
        print('Node features for the functional data was loaded.')
        return all_node_feats

    print('Creating and serializing for the structural data...')
    node2vec_emb_dir = join(os.getcwd(), os.pardir, 'PartIIProject', 'node2vec_embeds')
    all_node_feats = {}
    feats_limits = {}
    feat_size = 0
    for embed in os.listdir(node2vec_emb_dir):

        with open(join(node2vec_emb_dir, embed), 'r') as handle:

            format = handle.readline()
            nr_nodes = int(format.split()[0])
            feat_size = int(format.split()[1])
            graph_feats = np.zeros((nr_nodes, feat_size))
            for _ in range(nr_nodes):
                node_str_feat = handle.readline().split()
                curr_node = int(node_str_feat[0]) - 1
                for feat_index in range(feat_size):
                    graph_feats[curr_node][feat_index] = float(node_str_feat[feat_index + 1])
                    if feat_index not in feats_limits.keys():
                        feats_limits[feat_index] = [float(node_str_feat[feat_index + 1])]
                    else:
                        feats_limits[feat_index].append(float(node_str_feat[feat_index + 1]))

            all_node_feats[embed.split('embeddings')[0]] = graph_feats
    limits = [(min(feats_limits[feat_index]), max(feats_limits[feat_index])) for feat_index in range(feat_size)]
    for subj in all_node_feats.keys():
        for node_vect in all_node_feats[subj]:
            for feat_index in range(feat_size):
                node_vect[feat_index] = rescale_feats(limits[feat_index][0],
                                                      limits[feat_index][1],
                                                      node_vect[feat_index])
    with open(node_feats_file, 'wb') as handle:
        pkl.dump(all_node_feats, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print('Node features for the functional data was computed and persisted on disk.')

    return all_node_feats


def load_funct_data(model_GAT_choice):
    dataset_binary = join(ptnMAT_colab, 'dataset.pkl')
    if os.path.exists(dataset_binary):
        print('Loading the serialized data for the functional graphs...')
        with open(dataset_binary, 'rb') as handle:
            data = pkl.load(handle)
        print('Data set for the functional graphs was loaded.')
        return data['data'], data['subjs']

    dict_adj = get_functional_adjs()
    dict_node_feat = get_functional_node_feat()
    dict_tiv_score = get_NEO5_scores(model_GAT_choice.pers_traits)

    dict_dataset = {}
    available_subjs = []
    subjects = sorted(list(dict_adj.keys()))
    for subj_id in subjects:
        if subj_id in dict_node_feat.keys() and subj_id in dict_tiv_score.keys():
            dict_dataset[subj_id] = {}
            dict_dataset[subj_id]['feat'] = exp_dims(np.array(dict_node_feat[subj_id]), axis=0)
            dict_dataset[subj_id]['adj'] = exp_dims(np.array(dict_adj[subj_id]), axis=0)
            dict_dataset[subj_id]['bias'] = exp_dims(adj_to_bias(np.array(dict_adj[subj_id]), nhood=1), axis=0)
            dict_dataset[subj_id]['score'] = exp_dims(np.array(dict_tiv_score[subj_id]), axis=0)
            available_subjs.append(subj_id)

    with open(dataset_binary, 'wb') as handle:
        pkl.dump({'data': dict_dataset, 'subjs': sorted(available_subjs)}, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print('Data set for the functional graphs was computed and persisted on disk.')

    return dict_dataset, available_subjs
