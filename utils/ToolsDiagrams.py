import matplotlib.pyplot as plt
import seaborn as sns
from gat_impl.ConfigGAT import ConfigGAT
from baseline_impl.ConfigBaselines import ConfigBaselines
from Evaluation import outer_evaluation, get_best_models
from utils.LoadStructuralData import get_structural_adjacency, load_struct_data
from utils.LoadFunctionalData import get_functional_adjacency, load_funct_data
from gat_impl.InnerEvaluationGAT import inner_losses_gat
from baseline_impl.InnerEvaluationBaselines import inner_losses_baseline
from utils.ToolsDataProcessing import *
import math
import pandas as pd
import pickle as pkl

# Diagrams directory
gat_model_stats = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Diagrams')
if not os.path.exists(gat_model_stats):
    os.makedirs(gat_model_stats)


def plot_learning_history(model_gat_config: ConfigGAT) -> None:
    '''
     Plots the learning history of a GAT model
    :param model_gat_config: the hyperparameter configuration fo the specific model
    :return: void
    '''
    print("Restoring training logs from file %s." % model_gat_config.logs_file())
    with open(model_gat_config.logs_file(), 'rb') as logs_binary:
        logs = pkl.load(logs_binary)
    tr_loss, vl_loss = logs['history']['loss'], logs['history']['val_loss']
    nb_epochs = len(tr_loss)
    # Create data
    df = pd.DataFrame({'epoch': list(range(1, nb_epochs + 1)), 'train': np.array(tr_loss), 'val': np.array(vl_loss)})
    plt.plot('epoch', 'train', data=df, color='blue', label='training loss', linewidth=2.0)
    plt.plot('epoch', 'val', data=df, color='green', label='validation loss', linewidth=2.0)
    plt.xlabel('#epochs')
    plt.ylabel('MSE loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(gat_model_stats, 'loss_plot_' + str(model_gat_config) + '.pdf'))
    plt.show()


def plot_pq_ratio(model_gat_config: ConfigGAT) -> None:
    '''
     Plots the PQ ration of the custom early stopping from the training of a GAT model
    :param model_gat_config: the hyperparameter configuration fo the specific model
    :return: void
    '''
    print("Restoring training logs from file %s." % model_gat_config.logs_file())
    logs = {}
    for out_fold in range(model_gat_config.params['k_outer']):
        model_gat_config.params['eval_fold_out'] = out_fold
        model_gat_config.params['nested_CV_level'] = 'outer'
        if os.path.exists(model_gat_config.logs_file()):
            with open(model_gat_config.logs_file(), 'rb') as in_file:
                logs[out_fold] = np.array(pkl.load(in_file)['early_stop']['pq_ratio'])
        else:
            print('Missing results on the outer fold %d' % out_fold)
            continue
    logs['epoch'] = list(range(1, min(map(len, list(logs.values()))) + 1))
    # Create data
    df = pd.DataFrame(logs)
    colours = ['b', 'r', 'c', 'm', 'y']
    for out_split, colour in zip(range(model_gat_config.params['k_outer']), colours):
        plt.plot('epoch', 'split #' + str(out_split), data=df, color=colour, label='split #' + str(out_split),
                 linewidth=1.0)
    plt.title('PQ ratio history')
    plt.xlabel('#epochs')
    plt.ylabel('PQ-ratio value')
    plt.legend(loc='upper left')
    plt.show()


def plot_edge_weight_distribution(log_scale=10, data_name='structural'):
    '''
     Plots the distribution of the edge weight across a specific dataset
    :param log_scale: the log scale of the histogram
    :param data_name: the name of the dataset for the particular adjacency matrices
    :return: void
    '''
    if data_name == 'structural':
        adjacency_loader = get_structural_adjacency
    elif data_name == 'functional':
        adjacency_loader = get_functional_adjacency
    else:
        raise ValueError('Possible datasets: {structural,functional}, not %s' % data_name)

    # log-scale values of the weights, excluding the 0 edges
    dict_adj = adjacency_loader()
    all_edge_weights = np.concatenate([dict_adj[subj].flatten() for subj in dict_adj.keys()]).flatten()
    edge_weights = np.array([math.log(x, log_scale) for x in all_edge_weights if x != 0])
    lower_conf = np.percentile(edge_weights, 10)
    n, bins, patches = plt.hist(x=edge_weights, bins='auto', color='#0504aa', alpha=0.7, rwidth=None)
    for index, patch in enumerate(patches):
        patch.set_facecolor('blue' if lower_conf <= bins[index] else 'black')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Log%d edge weight value' % log_scale)
    plt.ylabel('Frequency')
    plt.title('Distribution of edge weights in %s graphs' % data_name)
    max_freq = n.max()
    # Set a y-axis limit.
    plt.ylim(top=np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10)
    plt.savefig(os.path.join(gat_model_stats, 'edge_weight_distrib_%s.pdf' % data_name))
    plt.show()


def plot_node_degree_distribution(get_adjs_loader=get_functional_adjacency, filter_flag=False):
    '''
     Plots the degree distribution in the graph population of a dataset
    :param get_adjs_loader: load specific adjacency matrices
    :param filter_flag: apply the threshold filtering
    :return: void
    '''
    data_type = get_adjs_loader.__name__.split('_')[1]
    adjs = np.array(list(get_adjs_loader().values()))
    if filter_flag:
        for i in range(len(adjs)):
            adjs[i] = lower_bound_filter(adjs[i])

    binary_adjs = [get_binary_adj(g) for g in adjs]
    degrees = np.array([[np.sum(curr_node_edges) for curr_node_edges in g] for g in binary_adjs]).flatten()
    n, bins, patches = plt.hist(x=degrees, bins='auto', color='#0504aa', alpha=0.7, rwidth=None)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Node degree value')
    plt.ylabel('Frequency')
    plt.title('Distribution of functional node degrees')
    mean = np.mean(degrees)
    var = np.var(degrees)
    plt.text(x=5, y=n.max() * 0.45, s=r'$\mu=%.2f, b=%.2f$' % (mean, var))
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(os.path.join(gat_model_stats, 'node_degrees_%s_filtered_%r.pdf' % (data_type, filter_flag)))
    plt.show()


def plot_pers_scores_distribution():
    '''
     Plots the distribution of the scores for each individual personality trait targeted
    :return: void
    '''
    scores_data = get_NEO5_scores(ConfigGAT().params['pers_traits_selection'])
    all_traits = np.array(list(scores_data.values())).transpose()
    trait_names = ConfigGAT().params['pers_traits_selection']
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
    '''
     Plot the error bars for the MSE loss on each inner evaluation and choice of hyperparameter value
    :param hyper_param: name of the hyperparameter
    :param model_name: the base name for the regression model: GAT, LR, SVR, RVM
    :return: void
    '''
    sns.set(style="whitegrid")
    model_type = ConfigGAT if model_name == 'GAT' else ConfigBaselines
    hyper_values = model_type.get_sampled_models(baseline_name=model_name)[hyper_param]
    losses = {}
    # retrieve the inner losses per specific value of the hyper-parameters to be plotted
    for hyper_value in hyper_values:
        if model_name == 'GAT':
            losses[hyper_value], _ = inner_losses_gat(filter_by_params={hyper_param: hyper_value})
        else:
            losses[hyper_value], _ = inner_losses_baseline(baseline_name=model_name,
                                                           filter_by_params={hyper_param: hyper_value})

    # use the eval loss on entire trait space: need to average the scores obtained for each individual trait
    for hyper_value, out_split_dict in losses.items():
        for out_split, model_dict in out_split_dict.items():
            for model, inner_split_dict in model_dict.items():
                for inner_split in inner_split_dict.keys():
                    inner_split_dict[inner_split] = np.mean(np.array(list(inner_split_dict[inner_split].values())))

    # for hyper-parameters that are not floats/ints, create a mapping to consecutive int values
    map_hyper_values = dict(zip(hyper_values, range(len(hyper_values))))
    # iterate the data and accumulate mean and stdev for each out_split and hyperparameter value combination
    colours = ['b', 'r', 'c', 'm', 'y']
    nb_choices = len(hyper_values)
    nb_out_splits = ConfigGAT().params['k_outer']
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
            # don't include a label for an error bar more than once
            label = None
            if out_split not in labels:
                labels.add(out_split)
                label = 'inner level #%d' % out_split
            markers, caps, bars = plt.errorbar(x, y, yerr=yerr, fmt='none', ecolor=colour, label=label, elinewidth=3,
                                               capsize=10)
            for bar in bars:
                bar.set_alpha(1.0)
            for cap in caps:
                cap.set_alpha(1.0)
    for i, colour in enumerate(colours):
        plt.plot([map_hyper_values[hyper_value] for hyper_value in hyper_values], np.transpose(avg_median)[i],
                 color=colour, alpha=0.75, linewidth=1)
    # Adding legend to the plot
    avg_median = np.mean(avg_median, axis=-1)
    plt.plot([map_hyper_values[hyper_value] for hyper_value in hyper_values], avg_median, color='red', linewidth=3)
    ticks = hyper_values
    if callable(hyper_values[0]):
        ticks = list(map(lambda x: x.__name__.split('_')[1], hyper_values))
    plt.xticks(sorted(list(map_hyper_values.values())), ticks, fontsize=10)
    plt.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.0, 0.0, 0.5, 1.0))
    plt.ylabel('Evaluation Loss', fontsize=12)
    plt.xlabel('Hyperparameter Value', fontsize=12)
    plt.savefig(os.path.join(gat_model_stats, 'error_bars_%s_%s.pdf' % (hyper_param, model_name)))
    plt.show()


def plot_comparison():
    '''
     Plots the comparison plot for each dataset between the regression models
    :return: void
    '''
    sns.set(style="whitegrid")
    out_eval_folds = np.array(sorted(list(range(ConfigGAT().params['k_outer']))))
    data_sets = [load_struct_data, load_funct_data]
    # each till visual parameters
    colors = ['r', 'g', 'b', 'm']
    relative_width = [-0.2, -0.1, 0.0, 0.1]
    model_type = ['LR', 'RVM', 'SVR', 'GAT']
    for data_set in data_sets:
        labels = set([])
        for color, width, model in zip(colors, relative_width, model_type):
            out_losses = outer_evaluation(model)
            best_fold_loss = np.zeros(len(out_eval_folds))
            for trait in ConfigGAT().params['pers_traits_selection']:
                best_fold_loss += out_losses[model][data_set][trait]['loss']
            best_fold_loss /= ConfigGAT().params['k_outer']
            # don't include an label for an error bar more than once
            label = None
            if model not in labels:
                if model == 'LR':
                    labels.add('Ridge')
                    label = 'Ridge'
                else:
                    labels.add(model)
                    label = model
            plt.bar(out_eval_folds + width, best_fold_loss, width=0.1, label=label, color=color, align='center')
        plt.legend(loc='upper right', frameon=True)
        plt.xticks(out_eval_folds, list(map(lambda x: 'fold #%d' % x, out_eval_folds)))
        plt.ylim(30, 52.5)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.xlabel('Outer fold', fontsize=12)
        plt.savefig(os.path.join(gat_model_stats, 'comparison_%s.pdf' % data_set.__name__.split('_')[1]))
        plt.show()


def plot_residuals_gat(out_fold, data_set) -> None:
    '''
     Plot the residuals for the GAT model for each trait on a specific dataset and outer evaluation fold
    :param out_fold: ID of the outer fold
    :param data_set: loader function for the dataset
    :return: void
    '''
    sns.set(style="whitegrid")
    colours = ['b', 'r', 'c', 'm', 'y']
    for trait, color in zip(ConfigGAT().params['pers_traits_selection'], colours):
        best_gat = get_best_models(model_name='GAT', data_set=data_set, trait=trait)
        config, _ = best_gat[out_fold]
        # set the configuration object for the outer evaluation of the best inner model
        config.params['nested_CV_level'] = 'outer'
        config.params['eval_fold_out'] = out_fold
        config.params['eval_fold_in'] = 0
        results = config.get_results()['predictions']
        true_score, predicted_score = map(lambda x: np.array(x), zip(*results[trait]))
        ax = sns.residplot(true_score, predicted_score, lowess=True, color=color, scatter_kws={'s': 20})
        ax.set(xlabel='observations', ylabel='residuals')
        plt.savefig(os.path.join(gat_model_stats, 'residual_trait_%s_data_%s_fold_%d.pdf' % (
            trait.replace('NEO.', ''), data_set.__name__.split('_')[1], out_fold)))
        plt.show()


if __name__ == "__main__":
    plot_comparison()
