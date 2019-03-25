import matplotlib.pyplot as plt

import networkx as nx
from matplotlib import pyplot, patches
from CrossValidatedGAT import *

CONF_LIMIT = 2.4148


def plt_learn_proc(model_GAT_config: GAT_hyperparam_config) -> None:
    print("Restoring training logs from file %s." % model_GAT_config.logs_file())
    with open(model_GAT_config.logs_file(), 'rb') as in_file:
        logs = pickle.load(in_file)
    tr_loss, vl_loss = [], []
    for epoch in range(1, logs['last_tr_epoch']):
        tr_loss.append(logs['logs'][epoch]['tr_loss'])
        vl_loss.append(logs['logs'][epoch]['val_loss'])
    # Create data
    df = pd.DataFrame({'epoch': list(range(1, len(vl_loss) + 1)), 'train': np.array(tr_loss), 'val': np.array(vl_loss)})
    plt.plot('epoch', 'train', data=df, color='blue', label='training loss')
    plt.plot('epoch', 'val', data=df, color='orange', label='validation loss')
    plt.title(str(model_GAT_config))
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(gat_model_stats, 'loss_plot_' + str(model_GAT_config) + '.png'))
    plt.show()


def plot_edge_weight_hist(log_scale=10, get_adjs_loader=get_structural_adjs):
    data_type = get_adjs_loader.__name__.split('_')[1]
    # log-scale values of the weights, excluding the 0 edges (self-edges still have weigth 0)
    edge_weights = np.array([math.log(x, log_scale) for x in persist_ew_data(get_adjs_loader) if x != 0])
    lower_conf = np.percentile(edge_weights, 10)
    print('Tail limit is %.4f' % lower_conf)
    n, bins, patches = plt.hist(x=edge_weights, bins='auto', color='#0504aa', alpha=0.7, rwidth=None)
    for index, patch in enumerate(patches):
        patch.set_facecolor('#0504aa' if lower_conf <= bins[index] else 'black')

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Log %d edge weight value' % log_scale)
    plt.ylabel('Frequency')
    plt.title('Edge Weight Histogram for %s graphs' % data_type)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(join(gat_model_stats, 'edge_weight_distrib_%s.png' % data_type))
    plt.show()


def plot_node_degree_hist(get_adjs_loader=get_structural_adjs, filter_flag=True):
    data_type = get_adjs_loader.__name__.split('_')[1]
    adjs = np.array(list(get_adjs_loader().values()))
    if filter_flag:
        for i in range(len(adjs)):
            adjs[i] = lower_bound_filter(CONF_LIMIT, adjs[i])

    binary_adjs = [get_binary_adj(g) for g in adjs]
    degrees = np.array([[np.sum(curr_node_edges) for curr_node_edges in g] for g in binary_adjs]).flatten()
    n, bins, patches = plt.hist(x=degrees, bins='auto', color='#0504aa', alpha=0.7, rwidth=None)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Node degree value')
    plt.ylabel('Frequency')
    plt.title('Node degrees Histogram for %s data when filtering is %r' % (data_type, filter_flag))
    mean = np.mean(degrees)
    var = np.var(degrees)
    plt.text(x=5, y=n.max() * 0.5, s=r'$\mu=%.2f, b=%.2f$' % (mean, var))
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(join(gat_model_stats, 'node_degrees_%s_filtered_%r.png' % (data_type,filter_flag)))
    plt.show()


def plot_pers_scores_hist():
    scores_data, trait_names = get_NEO5_scores()
    all_traits = np.array(list(scores_data.values())).transpose()

    packed_scores = zip(all_traits, trait_names)
    for trait_vals, trait_n in packed_scores:
        n, bins, patches = plt.hist(x=trait_vals, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Peronality trait value')
        plt.ylabel('Frequency')
        plt.title('Histogram of the distribution of %s trait' % trait_n)
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()


def draw_adjacency_heatmap(adjacency_matrix):
    im = plt.imshow(adjacency_matrix, cmap='YlGn', interpolation='nearest', )
    plt.colorbar(im)
    plt.show()


if __name__ == "__main__":
    hu_choices = [[20, 20, 10]]
    ah_choices = [[3, 3, 2]]
    aggr_choices = [MainGAT.concat_feature_aggregator, MainGAT.average_feature_aggregator,
                    MainGAT.master_node_aggregator]
    include_weights = [True]
    pers_traits = [['NEO.NEOFAC_A']]
    batch_chocies = [2]
    load_choices = [load_struct_data]
    for load, hu, ah, agg, iw, p_traits, batch_size in product(load_choices, hu_choices, ah_choices, aggr_choices,
                                                               include_weights,
                                                               pers_traits, batch_chocies):
        for eval_out in range(5):
            dict_param = {
                'hidden_units': hu,
                'attention_heads': ah,
                'include_ew': iw,
                'readout_aggregator': agg,
                'load_specific_data': load,
                'pers_traits_selection': p_traits,
                'batch_size': batch_size,
                'eval_fold_in': 4,
                'eval_fold_out': eval_out,
                'k_outer': 5,
                'k_inner': 5,
                'nested_CV_level': 'outer'

            }
            model = GAT_hyperparam_config(dict_param)
            plt_learn_proc(model)

