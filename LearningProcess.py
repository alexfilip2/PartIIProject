import matplotlib.pyplot as plt

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


def plot_hist_ew(log_scale=10):
    # log-scale values of the weights, excluding the 0 edges (self-edges still have weigth 0)
    edge_weights = np.array([math.log(x, log_scale) for x in persist_ew_data() if x != 0])
    lower_conf = np.percentile(edge_weights, 5)
    upper_conf = np.percentile(edge_weights, 95)
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=edge_weights, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    for index, patch in enumerate(patches):
        patch.set_facecolor('#0504aa' if lower_conf <= bins[index] <= upper_conf else 'black')

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Log %d edge weight value' % log_scale)
    plt.ylabel('Frequency')
    plt.title('Edge Weights Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


def plot_node_degree_hist(log_scale=10):
    adjs = np.array(list(get_struct_adjs().values()))
    binary_adjs = [get_binary_adj(g) for g in adjs]
    degrees = np.array([[np.sum(curr_node_edges) for curr_node_edges in g] for g in binary_adjs]).flatten()
    n, bins, patches = plt.hist(x=degrees, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Node degree value')
    plt.ylabel('Frequency')
    plt.title('Node degrees Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


if __name__ == "__main__":
    hid_units = [64, 32, 16]
    n_heads = [4, 4, 6]
    aggregators = [concat_feature_aggregator, average_feature_aggregator]
    include_weights = [True, False]
    limits = [(0, 80000), (183, 263857), (0, 500000), (80000, 4000000)]
    for aggr, iw, limit in product(aggregators, include_weights, limits):
        model_GAT_config = GAT_hyperparam_config(hid_units=hid_units,
                                                 n_heads=n_heads,
                                                 nb_epochs=1500,
                                                 aggregator=aggr,
                                                 include_weights=iw,
                                                 filter_name='interval',
                                                 limits=limit,
                                                 dataset_type='struct',
                                                 lr=0.0001,
                                                 l2_coef=0.0005)

        plt_learn_proc(model_GAT_config)
