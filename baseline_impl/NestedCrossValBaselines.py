from baseline_impl.HyperparametersBaselines import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, KFold
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


def generate_splits_baseline(baseline_config: HyperparametersBaselines, data: dict, outer_flag=True):
    '''
     Splits the entire dataset provided for the Nested CV as specified by the baseline configuration object
    :param baseline_config: configuration object
    :param data: the entire formatted dataset
    :param outer_flag: the recursive function behaves differently when splitting for outer level NCV
    :return: training and evaluation datasets
    '''
    outer_tr_data, outer_ts_data = {}, {}
    if outer_flag:
        data_sz = len(data['ftr_in'])
        # flatten the input feature matrices
        data['ftr_in'] = np.reshape(data['ftr_in'], newshape=[data_sz, -1])
        data['score_in'] = np.squeeze(data['score_in'])
        k_split = baseline_config.params['k_outer']
        eval_fold = baseline_config.params['eval_fold_out']
    else:
        k_split = baseline_config.params['k_inner']
        eval_fold = baseline_config.params['eval_fold_in']

    for input_type in {'ftr_in', 'score_in'}:
        splitter_out = KFold(k_split)
        train_indices, test_indices = list(splitter_out.split(X=data[input_type]))[eval_fold]
        outer_tr_data[input_type] = data[input_type][train_indices]
        outer_ts_data[input_type] = data[input_type][test_indices]

    if baseline_config.params['nested_CV_level'] == 'inner' and outer_flag:
        inner_tr_data, inner_ts_data = generate_splits_baseline(baseline_config, outer_tr_data, outer_flag=False)
        return inner_tr_data, inner_ts_data
    else:
        return outer_tr_data, outer_ts_data


def evaluate_baseline(baseline_config: HyperparametersBaselines):
    '''
    Train and evaluate the baseline model specified by the baseline configuration object
    :param baseline_config: configuration object for the baseline model
    :return: void
    '''
    # check for already trained-model results
    results = baseline_config.get_results()
    if results:
        return results
    # load the data
    tr_data, ts_data = generate_splits_baseline(baseline_config, data=baseline_config.load_data())
    # define the baseline model
    estimator = baseline_config.params['model'](**(baseline_config.get_suitable_args()))
    # train it
    estimator.fit(X=tr_data['ftr_in'], y=tr_data['score_in'])
    # get its predictions
    predictions = estimator.predict(X=ts_data['ftr_in'])
    if len(predictions.shape) == 2:
        predictions = np.squeeze(predictions, axis=-1)
    assert predictions.shape == ts_data['score_in'].shape
    # compute its evaluation MSE loss on those predictions
    test_loss = mean_squared_error(predictions, ts_data['score_in'])
    print('Test loss for the model %s is %.2f: ' % (baseline_config, test_loss))
    # save the predictions, loss and hyper-parameter config object
    with open(baseline_config.results_file(), 'wb') as results_binary:
        trait, = baseline_config.params['pers_traits_selection']
        results = {'predictions': {trait: zip(ts_data['score_in'], predictions)},
                   'test_loss': {trait: test_loss},
                   'config': baseline_config}
        pkl.dump(results, results_binary)

    return results


def inner_nested_cv_baselines(baseline_name):
    '''
     Performs the inner CV for a specific baseline model using all its sampled configuration
    :param baseline_name: base name of the the baseline
    :return: dict of losses per inner CV, model configuration and trait as well as a lookup table to translate the
    baseline name into a configuration object.
    '''
    inner_results = {}
    lookup_table = {}
    # retrieve the hyper-parameter search space
    grid = ParameterGrid(HyperparametersBaselines.get_sampled_models(baseline_name))
    for eval_out in range(HyperparametersBaselines().params['k_outer']):
        inner_results[eval_out] = {}
        for hyper_params in grid:
            for eval_in in range(HyperparametersBaselines().params['k_inner']):
                # create the configuration object for the baseline
                config = HyperparametersBaselines(hyper_params)
                # update it for the particular inner CV
                config.params['nested_CV_level'] = 'inner'
                config.params['eval_fold_in'] = eval_in
                config.params['eval_fold_out'] = eval_out
                # accumulate its results
                model_name = config.get_name()
                if model_name not in inner_results[eval_out].keys():
                    inner_results[eval_out][model_name] = {}
                if eval_in not in inner_results[eval_out][model_name].keys():
                    inner_results[eval_out][model_name][eval_in] = {}
                trait, = config.params['pers_traits_selection']
                inner_results[eval_out][model_name][eval_in][trait] = evaluate_baseline(config)['test_loss'][trait]
                lookup_table[model_name] = config

    return inner_results, lookup_table


def inner_losses_baseline(baseline_name, filter_by_params: dict = {}):
    '''
     Apply the inner CV for all the baseline regressions if not trained/evaluated already and get the
     results of some of them using the same hyperparameters as specified by the filter dict.
    :param baseline_name: base baseline name
    :param filter_by_params: dict of hyperparameter name and its value
    :return: dict of inner losses for the filtered models
    '''
    inner_losses_file = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                                     'baseline_inner_eval_losses.pck')
    if os.path.exists(inner_losses_file):
        with open(inner_losses_file, 'rb') as handle:
            inner_results, lookup_table = pkl.load(handle)
    else:
        lookup_table = {}
        inner_results = {}
        for baseline in {'LR', 'RVM', 'SVR'}:
            baseline_inner, baseline_lookup = inner_nested_cv_baselines(baseline)
            inner_results[baseline] = baseline_inner
            lookup_table[baseline] = baseline_lookup
        with open(inner_losses_file, 'wb') as handle:
            pkl.dump((inner_results, lookup_table), handle)

    # extract only the evaluation results of the models with specific hyper-parameters
    baseline_inner = inner_results[baseline_name]
    baseline_lookup = lookup_table[baseline_name]
    for out_split in baseline_inner.keys():
        model_names = list(baseline_inner[out_split].keys())
        for model in model_names:
            if not filter_by_params.items() <= baseline_lookup[model].params.items():
                baseline_inner[out_split].pop(model)

    return baseline_inner, baseline_lookup


if __name__ == "__main__":
    inner_nested_cv_baselines(baseline_name='RVM')
    inner_nested_cv_baselines(baseline_name='SVR')
    inner_nested_cv_baselines(baseline_name='LR')
    inner_losses_baseline('SVR')
