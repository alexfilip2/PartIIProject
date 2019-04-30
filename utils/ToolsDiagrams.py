import matplotlib.pyplot as plt
import seaborn as sns
from gat_impl.HyperparametersGAT import *
from baseline_impl.HyperparametersBaselines import *
from utils.Evaluation import outer_evaluation_gat, outer_evaluation_baselines
import math
import re
import networkx as nx

CONF_LIMIT = 2.4148
# Output of the learning process losses directory
gat_model_stats = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Diagrams')
if not os.path.exists(gat_model_stats):
    os.makedirs(gat_model_stats)


def plot_learning_history(model_gat_config: HyperparametersGAT) -> None:
    print("Restoring training logs from file %s." % model_gat_config.logs_file())
    with open(model_gat_config.logs_file(), 'rb') as logs_binary:
        logs = pkl.load(logs_binary)
    tr_loss, vl_loss = logs['history']['loss'], logs['history']['val_loss']
    nb_epochs = len(tr_loss)
    # Create data
    df = pd.DataFrame({'epoch': list(range(1, nb_epochs + 1)), 'train': np.array(tr_loss), 'val': np.array(vl_loss)})
    plt.plot('epoch', 'train', data=df, color='blue', label='training loss', linewidth=1.0)
    plt.plot('epoch', 'val', data=df, color='green', label='validation loss', linewidth=1.0)
    # plt.title(str(model_GAT_config))
    plt.xlabel('#epochs')
    plt.ylabel('MSE loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(gat_model_stats, 'loss_plot_' + str(model_gat_config) + '.pdf'))
    plt.show()


def plot_residuals(model_gat_config: HyperparametersGAT) -> None:
    sns.set(style="whitegrid")
    print("Restoring prediction results from file %s." % model_gat_config.results_file())
    with open(model_gat_config.results_file(), 'rb') as results_binary:
        results = pkl.load(results_binary)
    for trait in model_gat_config.params['pers_traits_selection']:
        true_score, predicted_score = map(lambda x: np.array(x), zip(*results[trait]))
        plt.figure()
        ax = sns.residplot(true_score, predicted_score, lowess=True, color="g")
        ax.set(xlabel='observations', ylabel='residuals')

        plt.show()


def plot_pq_ratio(model_gat_config: HyperparametersGAT) -> None:
    params = model_gat_config.params
    mode_specs = (params['load_specific_data'].__name__.split('_')[1], params['include_ew'],
                  params['readout_aggregator'].__name__.split('_')[0], params['batch_size'])
    save_fig_path = os.path.join(gat_model_stats, 'pq_ratio_plot_' + '_'.join(mode_specs) + '.pdf')
    if os.path.exists(save_fig_path):
        return
    print("Restoring training logs from file %s." % model_gat_config.logs_file())
    logs = {}
    for out_split in range(model_gat_config.params['k_outer']):
        model_gat_config.params['eval_fold_out'] = out_split
        if os.path.exists(model_gat_config.logs_file()):
            with open(model_gat_config.logs_file(), 'rb') as in_file:
                logs[out_split] = np.array(pkl.load(in_file)['early_stop']['pq_ratio'])
        else:
            continue
    logs['epoch'] = list(range(1, min(map(len, list(logs.values()))) + 1))
    # Create data
    df = pd.DataFrame(logs)
    colours = ['b', 'r', 'c', 'm', 'y']
    for out_split, colour in zip(range(model_gat_config.params['k_outer']), colours):
        plt.plot('epoch', 'split #' + str(out_split), data=df, color=colour, label='split #' + str(out_split),
                 linewidth=1.0)
    plt.title('PQ ratio: dataset %s, readout %s, include edges %s, batch size %s' % mode_specs)
    plt.xlabel('#epochs')
    plt.ylabel('pq_ratio')
    plt.legend(loc='upper left')
    plt.savefig(save_fig_path)
    plt.show()


def plot_edge_weight_distribution(log_scale=10, get_adjs_loader=get_structural_adjacency()):
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


def plot_node_degree_distribution(get_adjs_loader=get_structural_adjacency(), filter_flag=True):
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


def plot_pers_scores_distribution():
    scores_data = get_NEO5_scores(HyperparametersGAT().params['pers_traits_selection'])
    all_traits = np.array(list(scores_data.values())).transpose()
    trait_names = HyperparametersGAT().params['pers_traits_selection']
    packed_scores = zip(all_traits, trait_names)
    for trait_vals, trait_n in packed_scores:
        n, bins, patches = plt.hist(x=trait_vals, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Personality Trait Value')
        plt.ylabel('Population Frequency')
        plt.title('Distribution of the %s trait' % trait_n)
        max_freq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(top=np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10)
        plt.savefig(os.path.join(gat_model_stats, 'distribution_%s.pdf' % trait_n.split('.')[-1]))
        plt.show()


def plot_error_ncv(hyper_param, model_name):
    if model_name == 'GAT':
        model_type = HyperparametersGAT
    else:
        model_type = HyperparametersBaselines
    hyper_values = model_type.get_sampled_models(baseline_name=model_name)[hyper_param]
    losses = {}
    # retrieve the inner losses per specific value of the hyper-parameters to be plotted
    # losses is now a dict with levels: hyper_value -> out_split -> model -> inner_split -> trait_test_loss
    for hyper_value in hyper_values:
        losses[hyper_value], _ = model_type.inner_losses(
            filter_by_params={hyper_param: hyper_value, 'name': model_name})

    # use the eval loss on entire trait space: need to average the scores obtained for each individual trait
    for hyper_value, out_split_dict in losses.items():
        for out_split, model_dict in out_split_dict.items():
            for model, inner_split_dict in model_dict.items():
                for inner_split in inner_split_dict.keys():
                    inner_split_dict[inner_split] = np.mean(np.array(list(inner_split_dict[inner_split].values())))

    # for hyper-parameters that are not floats/ints, create a mapping to consecutive int values
    map_hyper_values = dict(zip(hyper_values, range(len(hyper_values))))

    # iterate the data and accumulate mean's and sdev's for each out_split x hyparam_value combination
    colours = ['b', 'r', 'c', 'm', 'y']
    nb_choices = len(hyper_values)
    nb_out_splits = len(list(losses[list(losses.keys())[0]].keys()))
    avg_median = np.zeros((nb_choices, nb_out_splits))
    labels = set([])
    for i, hyper_value in enumerate(hyper_values):
        for j, (colour, out_split) in enumerate(list(zip(colours, losses[hyper_value].keys()))):
            # gen all the losses of this inner cv averaged by the nr. of inner folds
            avg_loss_models = np.array([np.mean(np.array(list(m_dict.values()))) for model, m_dict in
                                        losses[hyper_value][out_split].items()])
            x = map_hyper_values[hyper_value]
            y = np.mean(avg_loss_models)
            yerr = np.std(avg_loss_models)
            avg_median[i][j] = y
            # don't include an label for an error bar more than once
            label = None
            if out_split not in labels:
                labels.add(out_split)
                label = 'Outer Split %d' % out_split
            markers, caps, bars = plt.errorbar(x, y, yerr=yerr, fmt='none', ecolor=colour, label=label, elinewidth=3,
                                               capsize=10)
            for bar in bars:
                bar.set_alpha(0.5)
            for cap in caps:
                cap.set_alpha(0.5)

    for j, colour in enumerate(colours):
        plt.plot([map_hyper_values[hyper_value] for hyper_value in hyper_values], np.transpose(avg_median)[j],
                 color=colour, alpha=0.5,
                 linewidth=1)
    # Adding legend to the plot
    avg_median = np.mean(avg_median, axis=-1)
    plt.plot([map_hyper_values[hyper_value] for hyper_value in hyper_values], avg_median, color='red', linewidth=2)
    ticks = hyper_values
    if callable(hyper_values[0]):
        ticks = list(map(lambda x: x.__name__, hyper_values))
    plt.xticks(sorted(list(map_hyper_values.values())), ticks)
    plt.legend(loc='best', frameon=True)
    plt.ylabel('Evaluation Loss', fontsize=14)
    plt.xlabel('Dropout rate ', fontsize=12)
    plt.show()


def plot_comparison():
    out_gat = outer_evaluation_gat()
    out_baselines = outer_evaluation_baselines()
    out_losses = {**out_gat, **out_baselines}
    sampled_hyper = HyperparametersGAT.get_sampled_models()
    out_eval_folds = np.array(sorted(list(HyperparametersBaselines().params['k_outer'].keys())))
    data_sets = sampled_hyper['load_specific_data']
    # each till visual paramaeters
    colors = ['r', 'g', 'y', 'b']
    relative_width = [-0.2, -0.1, 0.1, 0.2]
    model_type = ['GAT', 'RVM', 'LR', 'SVR']
    for data_set in data_sets:
        for trait in HyperparametersGAT().params['pers_traits_selection']:
            for color, width, model in zip(colors, relative_width, model_type):
                best_fold_loss = np.zeros(len(out_eval_folds))
                for i, eval_fold in enumerate(out_eval_folds):
                    best_fold_loss[i] = out_losses[model][data_set][eval_fold][trait]
                plt.bar(out_eval_folds + width, best_fold_loss, width=0.1, color=color, align='center')

            plt.xticks(out_eval_folds), list(map(lambda x: 'fold #%d' % x, out_eval_folds))
            plt.ylim(0, 60)
            plt.show()


if __name__ == "__main__":
    plot_error_ncv(model_name='GAT', hyper_param='load_specific_data')
