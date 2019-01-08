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
    feat_vect = [random.uniform(0, 1) for _ in range(10)]
    for nb_nodes in nb_nodes_graphs:
        graph_feats = [feat_vect for _ in range(nb_nodes)]
        features.append(graph_feats)
    return np.array(features)


def load_funct_data():
    adj_matrices = get_functional_adjs(PTN_MAT_DIM, sess_file=ptnMAT_d50_ses1)
    graph_features = gen_random_features([adj.shape[0] for adj in adj_matrices])
    pers_scores = get_NEO5_scores()[:adj_matrices.shape[0]]

    return adj_matrices, graph_features, pers_scores
