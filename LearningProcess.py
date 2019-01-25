import matplotlib.pyplot as plt

import networkx as nx
from matplotlib import pyplot, patches
from SupervisedGAT import *


def plt_learn_proc(model_GAT_config):
    train_losses_file = os.path.join(gat_model_stats, 'train_losses' + str(model_GAT_config))
    tr_loss, vl_loss = [], []
    with open(train_losses_file, 'r') as tr_loss_handle:
        for index, line in enumerate(tr_loss_handle, 1):
            tr_loss.append(float(line.split()[0]))
            vl_loss.append(float(line.split()[1]))
    # Create data
    df = pd.DataFrame({'epoch': list(range(1, index + 1)), 'train': np.array(tr_loss), 'val': np.array(vl_loss)})

    plt.plot('epoch', 'train', data=df, color='green', label='training loss')
    plt.plot('epoch', 'val', data=df, color='red', label='validation loss')
    plt.title(str(model_GAT_config))
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(gat_model_stats, 'loss_plot_' + str(model_GAT_config) + '.png'))
    plt.show()


def n_degree_empirical_distrib(hop=10000):
    edge_weights = list(map(int, persist_ew_data()))
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
    hid_units = [40, 20, 10]
    n_heads = [3, 3, 2]
    aggregators = [MainGAT.concat_feature_aggregator]
    include_weights = [True]
    limits = [(10000, 6000000)]
    pers_traits = [['A']]
    batches = [1]
    for aggr, iw, limit, p_traits, batch_size in product(aggregators, include_weights, limits, pers_traits, batches):
        model_GAT_config = GAT_hyperparam_config(hid_units=hid_units,
                                                 n_heads=n_heads,
                                                 nb_epochs=10000,
                                                 aggregator=aggr,
                                                 include_weights=iw,
                                                 filter_name='interval',
                                                 pers_traits=p_traits,
                                                 limits=limit,
                                                 batch_sz=batch_size,
                                                 dataset_type='struct',
                                                 lr=0.00001,
                                                 l2_coef=0.0005)
        plt_learn_proc(model_GAT_config)



