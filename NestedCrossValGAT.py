from gat_impl.ExecuteGAT import *
from utils.LoadFunctionalData import load_funct_data
from utils.LoadStructuralData import load_struct_data
from gat_impl.HyperparametersGAT import gat_result_dir
from sklearn.model_selection import ParameterGrid
import itertools
import math
import pickle as pkl
from itertools import product
import random


def sorted_stratification(gat_config, data_dict, unbalanced_sub, nesting_level):
    # identify the particular folds determined by the configuration of the model
    if gat_config.params['nested_CV_level'] == 'outer':
        k_split = gat_config.params['k_outer']
        eval_fold = gat_config.params['eval_fold_out']
    else:
        k_split = gat_config.params['k_inner']
        eval_fold = gat_config.params['eval_fold_in']
    # check if the split was already generated and present on disk and load if it is so
    split_id = 'split_%s_%d_%d_%d_%d_%s' % (nesting_level, gat_config.params['k_outer'], gat_config.params['k_inner'],
                                            gat_config.params['eval_fold_out'], gat_config.params['eval_fold_in'],
                                            ''.join(gat_config.params['pers_traits_selection']).replace('NEO.NEOFAC_',
                                                                                                        ''))

    split_file = os.path.join(gat_config.proc_data_dir(), split_id + '.pck')
    if os.path.exists(split_file):
        print('Reload the split of sorted stratification for the model %s' % gat_config)
        with open(split_file, 'rb') as split_binary:
            stratified_sub = pickle.load(split_binary)
    else:
        from random import randint
        # sort the subject ID's by their attached personality scores
        sorted_subjects = sorted(unbalanced_sub, key=lambda subj_name: sum(data_dict[subj_name]['score_in']))
        stratified_sub = [[] for _ in range(k_split)]
        # slide a window thought the subjects and assign them randomly to a fold
        for window_nr in range(len(sorted_subjects) // k_split):
            window = sorted_subjects[window_nr * k_split:(window_nr + 1) * k_split]
            assert len(window) == k_split
            for fold in range(k_split):
                random_index_window = randint(0, len(window) - 1)
                stratified_sub[fold].append(window[random_index_window])
                del window[random_index_window]
        # dump the rest of examples uniformly at random to the folds constructed so far
        for unassigned_elem in range(len(sorted_subjects) // k_split * k_split, len(sorted_subjects)):
            dump_fold_id = randint(0, k_split - 1)
            stratified_sub[dump_fold_id].append(sorted_subjects[unassigned_elem])
        # save the randomized split on disk
        with open(split_file, 'wb') as split_binary:
            pickle.dump(stratified_sub, split_binary, protocol=pkl.HIGHEST_PROTOCOL)

    # retrieve the particular train/val/test sets of this split
    test_sub = stratified_sub.pop(eval_fold)
    # choose the fold before the test fold (circularly) as the validation one
    val_fold = eval_fold - 1
    val_sub = stratified_sub.pop(val_fold)
    # delete the val/test sets from the entire data and concatenate it into a list of training subjects
    train_sub = list(itertools.chain.from_iterable(stratified_sub))

    def assert_disjoint(sets):
        for set_1, set_2 in product(sets, sets):
            if set_1 is not set_2:
                assert set(set_1).isdisjoint(set(set_2))

    assert_disjoint([train_sub, val_sub, test_sub])
    return train_sub, val_sub, test_sub


def format_for_keras(data_dict, list_sub):
    # the number of nodes of each graph
    N = data_dict[list_sub[0]]['adj_in'].shape[-1]
    # the initial dimension F of each node's feature vector
    F = data_dict[list_sub[0]]['ftr_in'].shape[-1]
    # the number of personality traits targeted at once
    S = len(data_dict[list_sub[0]]['score_in'])

    dataset_sz = len(list_sub)
    keras_formatted = {'ftr_in': np.empty(shape=(dataset_sz, N, F), dtype=np.float32),
                       'bias_in': np.empty(shape=(dataset_sz, N, N), dtype=np.float32),
                       'adj_in': np.empty(shape=(dataset_sz, N, N), dtype=np.float32),
                       'score_in': np.empty(shape=(dataset_sz, S), dtype=np.float32)}

    for example_index, s_key in enumerate(list_sub):
        for input_type in keras_formatted.keys():
            keras_formatted[input_type][example_index] = data_dict[s_key][input_type]

    return (keras_formatted['ftr_in'], keras_formatted['adj_in'],
            keras_formatted['bias_in'], keras_formatted['score_in'])


def load_cv_data(gat_config):
    # load the entire data set into main memory
    data, subjects = gat_config.params['load_specific_data'](gat_config.params)
    # prepare the outer split subjects
    train_sub, val_sub, test_sub = sorted_stratification(unbalanced_sub=subjects,
                                                         data_dict=data,
                                                         gat_config=gat_config,
                                                         nesting_level='outer')
    # prepare the inner split subjects
    if gat_config.params['nested_CV_level'] == 'inner':
        inner_sub = list(itertools.chain.from_iterable([train_sub, val_sub]))
        train_sub, val_sub, test_sub = sorted_stratification(unbalanced_sub=inner_sub,
                                                             data_dict=data,
                                                             gat_config=gat_config,
                                                             nesting_level='inner')
    # format the data for compatibility with the Keras GAT model
    tr_data = format_for_keras(data, train_sub)
    vl_data = format_for_keras(data, val_sub)
    ts_data = format_for_keras(data, test_sub)
    return tr_data, vl_data, ts_data


def sample_hyper_params(max_samples=18000):
    sampled_models_file = os.path.join(os.path.join(os.path.dirname(__file__)), 'sampled_models.pck')
    if os.path.exists(sampled_models_file):
        with open(sampled_models_file, 'rb') as handle:
            hyparam_choices = pickle.load(handle)
            return ParameterGrid(hyparam_choices)
    choices = {
        'learning_rate': [0.005, 0.001, 0.0005, 0.0001],
        'decay_rate': [0.0005],
        'attn_drop': [0.0, 0.2, 0.4, 0.6, 0.8],
        'readout_aggregator': [GATModel.average_feature_aggregator, GATModel.master_node_aggregator,
                               GATModel.concat_feature_aggregator],
        'load_specific_data': [load_struct_data, load_funct_data],
        'include_ew': [True, False],
        'batch_size': [32]}
    models_so_far = np.prod(np.array([len(choices[x]) for x in choices.keys()])) * 25
    sampling_left = math.floor(max_samples / models_so_far)
    NO_LAYERS = 3
    sample_ah = list(itertools.product(range(3, 7), repeat=NO_LAYERS))
    sample_hu = list(itertools.product(range(12, 48), repeat=NO_LAYERS))

    def check_feat_expansion(ah_hu_choice):
        for i in range(1, NO_LAYERS - 1):
            if ah_hu_choice[0][i] * ah_hu_choice[1][i] > ah_hu_choice[0][i - 1] * ah_hu_choice[1][i - 1]:
                return False
        # the last GAT layer averages node features (no multiplication with no of attention heads)
        if ah_hu_choice[1][-1] > ah_hu_choice[0][-2] * ah_hu_choice[1][-2]:
            return False
        return True

    valid_ah_hu = set(filter(lambda ah_hu_choice: check_feat_expansion(ah_hu_choice),
                             list(itertools.product(sample_ah, sample_hu))))
    choices['arch_width'] = list(map(lambda x: [list(x[0]), list(x[1])], random.sample(valid_ah_hu, sampling_left)))
    with open(sampled_models_file, 'wb') as handle:
        pickle.dump(choices, handle)

    return ParameterGrid(choices)


# perform the Nested Cross Validation
def nested_cross_validation_gat(test_flag=True):
    if test_flag:
        param_grid = {'arch_width': [[[2, 3, 2], [20, 15, 10]]],
                      'readout_aggregator': [GATModel.average_feature_aggregator],
                      'load_specific_data': [load_funct_data],
                      'include_ew': [True],
                      'attn_drop': [0.4],
                      'learning_rate': [0.0001],
                      'decay_rate': [0.0005],
                      'batch_size': [8]}
        grid = ParameterGrid(param_grid)
    else:
        grid = sample_hyper_params()
    dict_param = {
        'k_outer': 5,
        'k_inner': 5,
        'nested_CV_level': 'inner'}
    gat_model_config = HyperparametersGAT(dict_param)
    for eval_out in range(dict_param['k_outer']):
        gat_model_config.params['eval_fold_out'] = eval_out
        for params in grid:
            params['attention_heads'] = params['arch_width'][0]
            params['hidden_units'] = params['arch_width'][1]
            for eval_in in range(dict_param['k_inner']):
                gat_model_config.params['eval_fold_in'] = eval_in
                gat_model_config.update(params)
                if os.path.exists(gat_model_config.checkpoint_file()):
                    continue
                tr_data, vl_data, ts_data = load_cv_data(gat_config=gat_model_config)
                model = GATModel(config=gat_model_config)
                model.fit(training_data=tr_data, validation_data=vl_data)
                model.evaluate(test_data=ts_data)


def gat_eval_losses():
    inner_results = {}
    for log_file in os.listdir(gat_result_dir):
        if log_file.startswith('logs_'):
            # load the results for a suitable model into main memory
            with open(os.path.join(gat_result_dir, log_file), 'rb') as logs_file:
                config = HyperparametersGAT(pkl.load(logs_file)['params'])
            with open(config.results_file(), 'rb') as out_result_file:
                results = pkl.load(out_result_file)
                model_name = config.get_name()
                outer_split = config.params['eval_fold_out']
                inner_split = config.params['eval_fold_in']
            if outer_split not in inner_results.keys():
                inner_results[outer_split] = {}
            if model_name not in inner_results[outer_split].keys():
                inner_results[outer_split][model_name] = {}
            # save the test losses for the particular inner split
            inner_results[outer_split][model_name][inner_split] = results['test_loss']

    # print out the best average test loss (as a progress ?/nr of splits)
    best_losses = {}
    for out_split in inner_results.keys():
        best_losses[out_split] = {}
        avg_loss = {}
        print('The results for the outer split of ID %s are: ' % out_split)
        print()
        for model in inner_results[out_split].keys():
            avg_loss[model] = {}
            for trait in HyperparametersGAT().params['pers_traits_selection']:
                trait_inner_losses = []
                for in_split in inner_results[out_split][model].keys():
                    if trait in inner_results[out_split][model][in_split]:
                        trait_inner_losses.append(inner_results[out_split][model][in_split][trait])
                avg_loss[model][trait] = np.mean(np.array(trait_inner_losses))

        for trait in HyperparametersGAT().params['pers_traits_selection']:
            sort_by_trait = list(sorted(avg_loss.keys(), key=lambda key: avg_loss[key][trait]))
            best_losses[out_split][trait] = (sort_by_trait[0], avg_loss[sort_by_trait[0]][trait])
            print(
                'The BEST average test loss for trait %s is  %.3f' % (trait, avg_loss[sort_by_trait[0]][trait]))
            print('The model achieving this score is %s' % sort_by_trait[0])

            for t, loss in avg_loss[sort_by_trait[0]].items():
                print(t, loss)


if __name__ == "__main__":
    gat_eval_losses()
