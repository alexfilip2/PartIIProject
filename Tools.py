import numpy as np

np.set_printoptions(threshold=np.nan)
import os
import pandas as pd
import random
import operator


class GAT_hyperparam_config(object):
    def __init__(self,
                 hid_units,
                 n_heads,
                 nb_epochs=1000,
                 pers_traits=None,
                 dataset_type='structural',
                 edge_w_limit=50000,
                 lr=0.0001,
                 l2_coef=0.0005):
        self.nb_epochs = nb_epochs
        self.n_heads = n_heads
        self.hid_units = hid_units
        self.pers_traits = pers_traits if pers_traits is not None else ['A', 'O', 'C', 'N', 'E']
        self.dataset_type = dataset_type
        self.edge_w_limit = edge_w_limit
        self.lr = lr
        self.l2_coef = l2_coef

    def __str__(self):
        name = 'GAT_%s_attHeads%s_hidUnits%s_traits_%s_ew%d' % (self.dataset_type,
                                                                ",".join(map(str, self.n_heads)),
                                                                ",".join(map(str, self.hid_units)),
                                                                "".join(map(str, self.pers_traits)),
                                                                int(self.edge_w_limit / 1000))
        return name


pers_scores = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'TIVscores',
                           '1016_HCP_withTIV_acorrected_USETHIS.xlsx')


def get_NEO5_scores(trait_choice=None):
    df = pd.ExcelFile(pers_scores).parse('Raw_data')
    tiv_scores = []
    if trait_choice is None:
        trait_names = ['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E']
    else:
        trait_names = trait_choice
    for trait in sorted(trait_names):
        tiv_scores.append(df[trait])
    subjects = map(str, list(df['Subject']))
    tiv_score_dict = dict(zip(subjects, np.array(tiv_scores).transpose().tolist()))
    return tiv_score_dict


# transform an adjacency matrix with edge weights into a binary adj matrix
def get_binary_adj(graph):
    bin_adj = np.empty(graph.shape)
    for i in range(graph.shape[0]):
        for j in range(graph.shape[0]):
            if graph[i][j] > 0:
                bin_adj[i][j] = 1.0
    return bin_adj


# get the bias matrices (used for MASKED ATTENTION) of all the adjacency graphs
def adj_to_bias(adjs, sizes, nhood=1):
    # nr of graphs
    nb_graphs = adjs.shape[0]
    # an empty matrix of the same shape as adj
    mt = np.empty(adjs.shape)
    # iterate all the graphs
    for g in range(nb_graphs):
        # crate an identity matrix of the same shape as the adj for current graph (it includes only self-loops)
        mt[g] = np.eye(adjs.shape[1])
        # create a adj matrix  to include nhood-hop neighbours
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (get_binary_adj(adjs[g]) + np.eye(adjs[g].shape[0])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    # conjugate the adj matrix of nhood neighbours
    return -1e9 * (1.0 - mt)


def shuffle_tr_data(unshuf_scores, unshuf_feats, unshuf_biases, unshuf_adjs, chunk_sz):
    assert chunk_sz == len(unshuf_scores)

    shuffled_data = list(zip(unshuf_scores,
                             unshuf_feats[:chunk_sz],
                             unshuf_biases[:chunk_sz],
                             unshuf_adjs[:chunk_sz]))
    random.shuffle(shuffled_data)
    shuf_score, shuf_feats, shuf_bises, shuf_adjs = map(np.array, zip(*shuffled_data))

    assert len(shuf_score) == len(shuf_feats) == len(shuf_bises) == len(shuf_adjs) == chunk_sz

    return shuf_score, shuf_feats, shuf_bises, shuf_adjs


def shuffle_tr_utest(unshuf_scores, unshuf_feats, unshuf_biases, unshuf_adjs, chunk_sz):
    for _ in range(50):
        shuf_score, shuf_feats, shuf_bises, shuf_adjs = shuffle_tr_data(unshuf_scores,
                                                                        unshuf_feats,
                                                                        unshuf_biases,
                                                                        unshuf_adjs,
                                                                        chunk_sz)

        if unshuf_scores.shape != shuf_score.shape: print("error in shuffling")
        for row_s in shuf_score:
            check = False
            for row in unshuf_scores:
                if set(row_s.tolist()) == set(row.tolist()): check = True
            if not check:
                print("error in shuffling")
                break
