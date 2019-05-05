from baseline_impl.HyperparametersBaselines import *
from gat_impl.HyperparametersGAT import HyperparametersGAT
from NestedCrossValGAT import generate_splits
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import numpy as np
import itertools
import random

np.set_printoptions(threshold=sys.maxsize)
cached_data = {}


def load_baseline_data(baseline_config: HyperparametersBaselines):
    entire_data = baseline_config.params['load_specific_data'](baseline_config.params)
    subjects = list(entire_data.keys())
    dict_baseline_data = {}
    for subj_id in sorted(subjects):
        if subj_id not in dict_baseline_data.keys():
            dict_baseline_data[subj_id] = {}
        dict_baseline_data[subj_id]['ftr_in'] = entire_data[subj_id]['ftr_in'].flatten()
        dict_baseline_data[subj_id]['score_in'] = entire_data[subj_id]['score_in']
    return dict_baseline_data, subjects


def format_for_baselines(dict_baseline_data, specific_subj):
    # the dimensionality of the concatenated node features for an example graph
    flatten_dim = len(dict_baseline_data[random.choice(specific_subj)]['ftr_in'])
    nb_examples = len(specific_subj)
    baseline_formatted = {'ftr_in': np.empty(shape=(nb_examples, flatten_dim), dtype=np.float32),
                          'score_in': np.empty(shape=nb_examples, dtype=np.float32)}
    for example_index, subj_id in enumerate(specific_subj):
        if subj_id not in dict_baseline_data.keys():
            print('Failed formatting for baselines: missing subject from the whole dataset')
        for input_key in baseline_formatted.keys():
            baseline_formatted[input_key][example_index] = dict_baseline_data[subj_id][input_key]

    return baseline_formatted


def load_cv_baseline_data(baseline_config: HyperparametersBaselines):
    global cached_data
    dataset = baseline_config.params['load_specific_data']
    trait, = baseline_config.params['pers_traits_selection']
    if dataset not in cached_data.keys():
        cached_data[dataset] = {}
    if trait not in cached_data[dataset].keys():
        # load the entire baseline data
        dict_baseline_data, subjects = load_baseline_data(baseline_config)
        cached_data[dataset][trait] = (dict_baseline_data, subjects)

    # retrieve the data
    dict_baseline_data, subjects = cached_data[dataset][trait]
    gat_config = HyperparametersGAT(baseline_config.params)
    # prepare the outer split subjects
    train_sub, val_sub, test_sub = generate_splits(unbalanced_sub=subjects,
                                                   data_dict=dict_baseline_data,
                                                   gat_config=gat_config,
                                                   nesting_level='outer')
    # prepare the inner split subjects
    if gat_config.params['nested_CV_level'] == 'inner':
        inner_sub = list(itertools.chain.from_iterable([train_sub, val_sub]))
        train_sub, val_sub, test_sub = generate_splits(unbalanced_sub=inner_sub,
                                                       data_dict=dict_baseline_data,
                                                       gat_config=gat_config,
                                                       nesting_level='inner')
    # format the data for compatibility with the Keras GAT model
    tr_data = format_for_baselines(dict_baseline_data, train_sub)
    ts_data = format_for_baselines(dict_baseline_data, test_sub)
    return tr_data, ts_data


def evaluate_baseline(baseline_config: HyperparametersBaselines):
    if os.path.exists(baseline_config.results_file()):
        with open(baseline_config.results_file(), 'rb') as fp:
            loss_result = pkl.load(fp)
            return loss_result
    tr_data, ts_data = load_cv_baseline_data(baseline_config)
    # create the ML model
    estimator = baseline_config.params['model'](**(baseline_config.get_suitable_args()))
    # train it on the specific folds
    estimator.fit(X=tr_data['ftr_in'], y=tr_data['score_in'])
    # get its predictions on the designated  inner evaluation fold
    predictions = estimator.predict(X=ts_data['ftr_in'])
    # compute the MSE loss on predicting that personality trait
    test_loss = mean_squared_error(predictions, ts_data['score_in'])
    print('Test loss for the model %s is %.2f: ' % (baseline_config, test_loss))
    # save the predictions, loss and hyper-parameter config object together
    with open(baseline_config.results_file(), 'wb') as results_binary:
        trait, = baseline_config.params['pers_traits_selection']
        results = {'predictions': {trait: predictions},
                   'test_loss': {trait: test_loss},
                   'config': baseline_config}
        pkl.dump(results, results_binary)
    return results


def nested_cross_validation_baselines(baseline_name):
    # the baseline model to be evaluated
    ncv_params = {'k_outer': HyperparametersBaselines().params['k_outer'],
                  'k_inner': HyperparametersBaselines().params['k_outer'],
                  'nested_CV_level': 'inner'}
    # retrieve the hyper-parameter search space
    grid = ParameterGrid(HyperparametersBaselines.get_sampled_models(baseline_name))
    for eval_out in range(ncv_params['k_outer']):
        ncv_params['eval_fold_out'] = eval_out
        for hyper_params in grid:
            for eval_in in range(ncv_params['k_inner']):
                ncv_params['eval_fold_in'] = eval_in
                # create the configuration object for the baseline
                config = HyperparametersBaselines(hyper_params)
                config.update(ncv_params)
                evaluate_baseline(baseline_config=config)


if __name__ == "__main__":
    nested_cross_validation_baselines('LR')
    nested_cross_validation_baselines('RVM')
    nested_cross_validation_baselines('SVR')
