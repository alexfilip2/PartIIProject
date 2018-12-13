from Tools import *

PTN_MAT_DIM = 50
ptnMAT_colab = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'functional_data')
ptnMAT_d50_dir = os.path.join(ptnMAT_colab, '3T_HCP1200_MSMAll_d50_ts2')

ptnMAT_d50_ses1 = os.path.join(ptnMAT_d50_dir, 'netmats1.txt')
ptnMAT_d50_ses2 = os.path.join(ptnMAT_d50_dir, 'netmats2.txt')


def get_adj_ses1(dim):
    adj = []
    with open(ptnMAT_d50_ses1, 'r', encoding='UTF-8') as data:
        for line in data:
            graph = [[0 for x in range(dim)] for y in range(dim)]
            for index, edge_weight in enumerate(line.split()):
                graph[int(index / dim)][int(index % dim)] = float(edge_weight)
            adj.append(graph)

    return np.array(adj)


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
    adj_matrices = get_adj_ses1(PTN_MAT_DIM)
    graphs_features = gen_random_features([adj.shape[0] for adj in adj_matrices])
    data_scores = get_NEO5_scores()[:adj_matrices.shape[0]]
    score_train, score_val, score_test = np.split(data_scores,
                                                  [int(len(data_scores) * 0.8), int(len(data_scores) * 0.9)])
    return adj_matrices, graphs_features, score_train, score_val, score_test
