import matplotlib.pyplot as plt
from ToolsStructural import *


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


def node_degree_distrib(limit):
    ew_file = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'flatten_edge_weigths.npy')
    if os.path.exists(ew_file):
        print('Loading the serialized edge weights data...')
        edge_weights = np.load(ew_file)
        print('Edge weights data was loaded.')
    else:
        print('Creating and serializing edge weights data...')
        edge_weights = np.array(list(get_struct_adjs().values())).flatten()

        np.save(ew_file, edge_weights)
        print('Edge weights data was persisted on disk.')

    edge_weights = list(map(int, edge_weights))
    print(max(edge_weights))

    filter_weights = [ew for ew in edge_weights if ew < limit]

    print('The ratio of elements under the limit %d is %f' % (limit, len(filter_weights) / len(edge_weights)))


def plot_hist_ew(only_conf_interv=True, log_scale=10):
    ew_file = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'flatten_edge_weigths.npy')
    if os.path.exists(ew_file):
        print('Loading the serialized edge weights data...')
        edge_weights = np.load(ew_file)
        print('Edge weights data was loaded.')
    else:
        print('Creating and serializing edge weights data...')
        adjs = list(get_struct_adjs().values())
        edge_weights = [np.array(mat)[np.triu_indices(len(mat))] for mat in adjs]
        edge_weights = np.array(edge_weights).flatten()
        np.save(ew_file, edge_weights)
        print('Edge weights data was persisted on disk.')
    # log-scale values of the weights, excluding the 0 edges (self-edges still have weigth 0)
    edge_weights = np.array([math.log(x, log_scale) for x in edge_weights if x != 0])

    lower_conf = np.percentile(edge_weights, 5)
    upper_conf = np.percentile(edge_weights, 95)
    if only_conf_interv:
        return lower_conf, upper_conf

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=edge_weights, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    for index, patch in enumerate(patches):
        patch.set_facecolor('#0504aa' if lower_conf <= bins[index] <= upper_conf else 'black')

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Edge weight value')
    plt.ylabel('Frequency')
    plt.title('Edge Weights Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

    return lower_conf, upper_conf


if __name__ == "__main__":
    '''
    hid_units = [64, 32, 16]
    n_heads = [4, 4, 6]
    edge_w_limits = [80000, 200000, 4000000]
    aggregators = [concat_feature_aggregator, average_feature_aggregator]
    include_weights = [True]
    for ew_limit, aggr, iw in product(edge_w_limits, aggregators, include_weights):
        model_GAT_config = GAT_hyperparam_config(hid_units=hid_units,
                                                 n_heads=n_heads,
                                                 nb_epochs=1500,
                                                 edge_w_limit=ew_limit,
                                                 aggregator=aggr,
                                                 include_weights=iw,
                                                 dataset_type='struct',
                                                 lr=0.0001,
                                                 l2_coef=0.0005)
        plt_learn_proc(model_GAT_config)
    '''
    print(plot_hist_ew(only_conf_interv=False))
