from gat_impl.HyperparametersGAT import *
from sklearn.model_selection import ParameterGrid
import itertools
import pickle as pkl
from itertools import product
import numpy as np
import os
from random import randint


def reload_splits(gat_config: HyperparametersGAT, nesting_level: str):
    # identify the particular folds determined by the configuration of the model
    if gat_config.params['nested_CV_level'] == 'outer':
        k_split, eval_fold = gat_config.params['k_outer'], gat_config.params['eval_fold_out']
    else:
        k_split, eval_fold = gat_config.params['k_inner'], gat_config.params['eval_fold_in']
    # check if the split was already generated and persisted on disk and load it
    split_id = 'split_%s_%d_%d_%d_%d_%s' % (nesting_level,
                                            gat_config.params['k_outer'],
                                            gat_config.params['k_inner'],
                                            gat_config.params['eval_fold_out'],
                                            gat_config.params['eval_fold_in'],
                                            gat_config.get_summarized_traits())

    saved_split_file = os.path.join(gat_config.processed_data_dir(), split_id + '.pck')
    if os.path.exists(saved_split_file):
        with open(saved_split_file, 'rb') as split_binary:
            stratified_sub = pkl.load(split_binary)
    else:
        stratified_sub = None

    return stratified_sub, k_split, eval_fold, saved_split_file


def generate_splits(gat_config: HyperparametersGAT, data_dict: dict, unbalanced_sub: list, nesting_level: str):
    stratified_sub, k_split, eval_fold, saved_split_file = reload_splits(gat_config, nesting_level)

    if stratified_sub is None:
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
        for unassigned_elem in range(window_nr, len(sorted_subjects)):
            dump_fold_id = randint(0, k_split - 1)
            stratified_sub[dump_fold_id].append(sorted_subjects[unassigned_elem])
        # save the randomized split on disk
        with open(saved_split_file, 'wb') as split_binary:
            pkl.dump(stratified_sub, split_binary, protocol=pkl.HIGHEST_PROTOCOL)

    # retrieve the particular train/val/test sets of this split
    test_sub = stratified_sub.pop(eval_fold)
    # choose the fold before the test fold (circularly) as the validation one
    val_fold = eval_fold - 1
    val_sub = stratified_sub.pop(val_fold)
    # delete the val/test sets from the entire data and concatenate it into a list of training subjects
    train_sub = list(itertools.chain.from_iterable(stratified_sub))

    def assert_disjoint(sets: list):
        for set_1, set_2 in product(sets, sets):
            if set_1 is not set_2:
                assert set(set_1).isdisjoint(set(set_2))

    assert_disjoint([train_sub, val_sub, test_sub])
    return train_sub, val_sub, test_sub


def format_for_keras(data_dict: dict, list_sub: list) -> tuple:
    # the number of nodes of each graph, dimension F of node feature, personality traits targeted at once
    N = data_dict[list_sub[0]]['adj_in'].shape[-1]
    F = data_dict[list_sub[0]]['ftr_in'].shape[-1]
    S = len(data_dict[list_sub[0]]['score_in'])
    dataset_sz = len(list_sub)
    formatted = {'ftr_in': np.empty(shape=(dataset_sz, N, F), dtype=np.float32),
                 'bias_in': np.empty(shape=(dataset_sz, N, N), dtype=np.float32),
                 'adj_in': np.empty(shape=(dataset_sz, N, N), dtype=np.float32),
                 'score_in': np.empty(shape=(dataset_sz, S), dtype=np.float32)}

    for example_index, s_key in enumerate(list_sub):
        for input_type in formatted.keys():
            formatted[input_type][example_index] = data_dict[s_key][input_type]

    return (formatted['ftr_in'], formatted['adj_in'], formatted['bias_in'], formatted['score_in'])


def load_cv_data(gat_config: HyperparametersGAT):
    # load the entire data set into main memory
    data = gat_config.params['load_specific_data'](gat_config.params)
    subjects = list(data.keys())
    # prepare the outer split subjects
    train_sub, val_sub, test_sub = generate_splits(unbalanced_sub=subjects, data_dict=data, gat_config=gat_config,
                                                   nesting_level='outer')
    # prepare the inner split subjects
    if gat_config.params['nested_CV_level'] == 'inner':
        inner_sub = list(itertools.chain.from_iterable([train_sub, val_sub]))
        train_sub, val_sub, test_sub = generate_splits(unbalanced_sub=inner_sub, data_dict=data, gat_config=gat_config,
                                                       nesting_level='inner')
    # format the data for compatibility with the Keras GAT model
    tr_data = format_for_keras(data, train_sub)
    vl_data = format_for_keras(data, val_sub)
    ts_data = format_for_keras(data, test_sub)
    return tr_data, vl_data, ts_data


def evaluate_gat(config: HyperparametersGAT):
    if os.path.exists(config.results_file()):
        with open(config.results_file(), 'rb') as fp:
            loss_result = pkl.load(fp)
            return loss_result
    tr_data, vl_data, ts_data = load_cv_data(gat_config=config)
    model = GATModel(config=config)
    model.fit(training_data=tr_data, validation_data=vl_data)
    return model.evaluate(test_data=ts_data)


# perform the Nested Cross Validation
def nested_cross_validation_gat():
    grid = ParameterGrid(HyperparametersGAT.get_sampled_models())
    ncv_params = {'k_outer': HyperparametersGAT().params['k_outer'],
                  'k_inner': HyperparametersGAT().params['k_inner'],
                  'nested_CV_level': 'inner'}
    for params in grid:
        for eval_out in range(ncv_params['k_outer']):
            for eval_in in range(ncv_params['k_inner']):
                # update the architecture hyper-parameters
                params['attention_heads'] = params['arch_width'][0]
                params['hidden_units'] = params['arch_width'][1]
                gat_model_config = HyperparametersGAT(params)
                # update the nested CV hyper-parameters
                ncv_params['eval_fold_out'] = eval_out
                ncv_params['eval_fold_in'] = eval_in
                gat_model_config.update(ncv_params)
                # train and evaluate the GAT model
                evaluate_gat(gat_model_config)


if __name__ == "__main__":
    nested_cross_validation_gat()
