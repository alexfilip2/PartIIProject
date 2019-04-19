import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import pyplot, patches
import seaborn as sns
from NestedCrossValGAT import *
import math
import re



CONF_LIMIT = 2.4148
# Output of the learning process losses directory
gat_model_stats = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Diagrams')
if not os.path.exists(gat_model_stats):
    os.makedirs(gat_model_stats)


def plt_learn_proc(model_GAT_config: HyperparametersGAT) -> None:
    print("Restoring training logs from file %s." % model_GAT_config.logs_file())
    with open(model_GAT_config.logs_file(), 'rb') as in_file:
        logs = pickle.load(in_file)
    tr_loss, vl_loss = logs['history']['loss'], logs['history']['val_loss']
    nb_epochs = len(tr_loss)
    # Create data
    df = pd.DataFrame({'epoch': list(range(1, nb_epochs + 1)), 'train': np.array(tr_loss), 'val': np.array(vl_loss)})
    plt.plot('epoch', 'train', data=df, color='blue', label='training loss', linewidth=1.0)
    plt.plot('epoch', 'val', data=df, color='green', label='validation loss', linewidth=1.0)
    #plt.title(str(model_GAT_config))
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(gat_model_stats, 'loss_plot_' + str(model_GAT_config) + '.pdf'))
    plt.show()


def plt_residuals(model_GAT_config: HyperparametersGAT) -> None:
    sns.set(style="whitegrid")
    print("Restoring prediction results from file %s." % model_GAT_config.results_file())
    with open(model_GAT_config.results_file(), 'rb') as in_file:
        results = pickle.load(in_file)
    for pers_trait in model_GAT_config.params['pers_traits_selection']:
        true_score, predicted_score = map(lambda x: np.array(x), zip(*results[pers_trait]))
        plt.figure()
        ax = sns.residplot(true_score, predicted_score, lowess=True, color="g")
        ax.set(xlabel='common xlabel', ylabel='common ylabel')

        plt.show()


def plot_pq_ratio(model_GAT_config: HyperparametersGAT) -> None:
    mode_specs = (model_GAT_config.params['load_specific_data'].__name__.split('_')[1],
                  model_GAT_config.params['readout_aggregator'].__name__.split('_')[0],
                  str(model_GAT_config.params['include_ew']),
                  str(model_GAT_config.params['batch_size']))
    save_fig_path = os.path.join(gat_model_stats, 'pq_ratio_plot_' + '_'.join(mode_specs) + '.pdf')
    if os.path.exists(save_fig_path): return
    print("Restoring training logs from file %s." % model_GAT_config.logs_file())
    logs = {}
    for out_split in range(model_GAT_config.params['k_outer']):
        model_GAT_config.update({'eval_fold_out': out_split})
        if os.path.exists(model_GAT_config.logs_file()):
            with open(model_GAT_config.logs_file(), 'rb') as in_file:
                logs['split_' + str(out_split)] = np.array(pickle.load(in_file)['early_stop']['pq_ratio'])
        else:
            return
    logs['epoch'] = list(range(1, min(map(len, list(logs.values()))) + 1))
    # Create data
    df = pd.DataFrame(logs)
    colours = ['b', 'r', 'c', 'm', 'y']
    for out_split, colour in zip(range(model_GAT_config.params['k_outer']), colours):
        plt.plot('epoch', 'split_' + str(out_split), data=df, color=colour, label='split_' + str(out_split),
                 linewidth=1.0)
    plt.title('PQ ratio: dataset %s, readout %s, include edges %s, batch size %s' % mode_specs)
    plt.xlabel('epoch')
    plt.ylabel('pq_ratio')
    plt.legend(loc='upper left')
    plt.savefig(save_fig_path)
    plt.show()


def plt_all_learn_curves(plot_funct):
    config = HyperparametersGAT()
    for file in sorted(os.listdir(checkpts_dir)):
        if file.startswith('logs_'):
            with open(os.path.join(checkpts_dir, file), 'rb') as checkpoint:
                true_config = pkl.load(checkpoint)['params']
                config.update(true_config)
                plot_funct(config)



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
    plt.savefig(os.path.join(gat_model_stats, 'edge_weight_distrib_%s.pdf' % data_type))
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
    plt.savefig(os.path.join(gat_model_stats, 'node_degrees_%s_filtered_%r.pdf' % (data_type, filter_flag)))
    plt.show()


def plot_pers_scores_hist():
    scores_data, trait_names = get_NEO5_scores(HyperparametersGAT().params)
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
    plt_all_learn_curves(plt_learn_proc)
