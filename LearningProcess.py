import matplotlib.pyplot as plt

import networkx as nx
from matplotlib import pyplot, patches
from CrossValidatedGAT import *


def plt_learn_proc(model_GAT_config: GAT_hyperparam_config) -> None:
    print("Restoring training logs from file %s." % model_GAT_config.logs_file())
    with open(model_GAT_config.logs_file(), 'rb') as in_file:
        logs = pickle.load(in_file)['logs']
    tr_loss, vl_loss = [],[]
    for epoch in range(1, model_GAT_config.params['num_epochs']):
        if epoch in logs.keys():
            tr_loss.append(logs[epoch]['tr_loss'])
            vl_loss.append(logs[epoch]['val_loss'])
        else:
            break
    # Create data
    df = pd.DataFrame({'epoch': list(range(1, len(vl_loss)+1)), 'train': np.array(tr_loss), 'val': np.array(vl_loss)})
    plt.plot('epoch', 'train', data=df, color='blue', label='training loss')
    plt.plot('epoch', 'val', data=df, color='orange', label='validation loss')
    plt.title(str(model_GAT_config))
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(gat_model_stats, 'loss_plot_' + str(model_GAT_config) + '.png'))
    plt.show()


def n_degree_empirical_distrib(hop=10000):
    edge_weights = list(map(int, persist_ew_data(get_adjs_loader=get_structural_adjs())))
    y_ratio, x_limit = [], []
    for limit in range(1, max(edge_weights), hop):
        filter_weights = [ew for ew in edge_weights if ew < limit]
        y_ratio.append(len(filter_weights) / len(edge_weights))
        x_limit.append(limit / hop)

    plt.plot(x_limit, y_ratio)
    plt.show()


def plot_edge_weight_hist(log_scale=10, get_adjs_loader=get_structural_adjs):
    data_type = get_adjs_loader.__name__.split('_')[1]
    # log-scale values of the weights, excluding the 0 edges (self-edges still have weigth 0)
    edge_weights = np.array([math.log(x, log_scale) for x in persist_ew_data(get_adjs_loader) if x != 0])
    lower_conf = np.percentile(edge_weights, 5)
    upper_conf = np.percentile(edge_weights, 95)
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=edge_weights, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    for index, patch in enumerate(patches):
        patch.set_facecolor('#0504aa' if lower_conf <= bins[index] <= upper_conf else 'black')

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Log %d edge weight value' % log_scale)
    plt.ylabel('Frequency')
    plt.title('Edge Weights Histogram for %s graphs' % data_type)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(join(gat_model_stats, 'edge_weight_distrib_%s.png' % data_type))
    plt.show()


def plot_node_degree_hist(log_scale=10, get_adjs_loader=get_structural_adjs):
    data_type = get_adjs_loader.__name__.split('_')[1]
    adjs = np.array(list(get_adjs_loader().values()))
    binary_adjs = [get_binary_adj(g) for g in adjs]
    degrees = np.array([[np.sum(curr_node_edges) for curr_node_edges in g] for g in binary_adjs]).flatten()
    n, bins, patches = plt.hist(x=degrees, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Node degree value')
    plt.ylabel('Frequency')
    plt.title('Node degrees Histogram for %s data' % data_type)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(join(gat_model_stats, 'node_degrees_filtered.png'))
    plt.show()


def plot_pers_scores_hist():
    all_traits = np.array(list(get_NEO5_scores().values())).transpose()

    for trait_vals in all_traits:
        n, bins, patches = plt.hist(x=trait_vals, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Peronality trait value')
        plt.ylabel('Frequency')
        plt.title('Peronality trait Histogram')
        plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()


def draw_adjacency_heatmap(adjacency_matrix):
    plt.imshow(adjacency_matrix, cmap='hot', interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    hid_units = [20, 20, 5]
    n_heads = [3, 3, 2]
    aggregators = [MainGAT.average_feature_aggregator]
    include_weights = [False]
    pers_traits = [['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E']]
    batches = [2]
    for aggr, iw, p_traits, batch_size in product(aggregators, include_weights, pers_traits, batches):
        dict_param = {
            'hidden_units': hid_units,
            'attention_heads': n_heads,
            'include_ew': iw,
            'readout_aggregator': aggr,
            'load_specific_data': load_struct_data,
            'pers_traits_selection': p_traits,
            'batch_size': batch_size,
            'edgeWeights_filter': None,
            'learning_rate': 0.0001,
        }
        plt_learn_proc(GAT_hyperparam_config(dict_param))
