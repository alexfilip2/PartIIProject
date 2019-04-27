from baseline_impl.HyperparametersBaselines import *
from gat_impl.HyperparametersGAT import HyperparametersGAT
from NestedCrossValGAT import sorted_stratification
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import numpy as np
import itertools


def load_baseline_data(baseline_config: HyperparametersBaselines):
    entire_data, subjects = baseline_config.params['load_specific_data'](baseline_config.params)
    assert set(list(entire_data.keys())) == set(subjects)
    dict_baseline_data = {}
    for subj_id in sorted(subjects):
        if subj_id not in dict_baseline_data.keys():
            dict_baseline_data[subj_id] = {}
        dict_baseline_data[subj_id]['ftr_in'] = entire_data[subj_id]['ftr_in'].flatten()
        dict_baseline_data[subj_id]['score_in'] = entire_data[subj_id]['score_in']
    return dict_baseline_data, subjects


def format_for_baselines(dict_baseline_data, specific_sub):
    # the dimensionality of the concatenated node features for an example graph
    flatten_dim = len(dict_baseline_data[random.choice(specific_sub)]['ftr_in'])
    dataset_sz = len(specific_sub)
    baseline_formatted = {'ftr_in': np.empty(shape=(dataset_sz, flatten_dim), dtype=np.float32),
                          'score_in': np.empty(shape=dataset_sz, dtype=np.float32)}
    for input_index, subj_id in enumerate(specific_sub):
        if subj_id not in dict_baseline_data.keys():
            print('Failed formatting for baselines: missing subject from the whole dataset')
        for input_key in baseline_formatted.keys():
            baseline_formatted[input_key][input_index] = dict_baseline_data[subj_id][input_key]

    return baseline_formatted


def load_cv_baseline_data(baseline_config: HyperparametersBaselines):
    dict_baseline_data, subjects = load_baseline_data(baseline_config)
    # ensure we evaluate the baselines on same splits as GAT (which predicts all traits at once)
    baseline_params = baseline_config.params.copy()
    baseline_params.pop('pers_traits_selection')
    gat_config = HyperparametersGAT(baseline_params)
    # prepare the outer split subjects
    train_sub, val_sub, test_sub = sorted_stratification(unbalanced_sub=subjects,
                                                         data_dict=dict_baseline_data,
                                                         gat_config=gat_config,
                                                         nesting_level='outer')
    # prepare the inner split subjects
    if gat_config.params['nested_CV_level'] == 'inner':
        inner_sub = list(itertools.chain.from_iterable([train_sub, val_sub]))
        train_sub, val_sub, test_sub = sorted_stratification(unbalanced_sub=inner_sub,
                                                             data_dict=dict_baseline_data,
                                                             gat_config=gat_config,
                                                             nesting_level='inner')
    # format the data for compatibility with the Keras GAT model
    tr_data = format_for_baselines(dict_baseline_data, train_sub)
    ts_data = format_for_baselines(dict_baseline_data, test_sub)
    return tr_data, ts_data


def nested_cross_validation_baselines(hyper_search_space):
    # the baseline model to be evaluated
    dict_param = {
        'k_outer': 5,
        'k_inner': 5,
        'nested_CV_level': 'inner'}
    grid = ParameterGrid(hyper_search_space)

    for eval_out in range(dict_param['k_outer']):
        dict_param['eval_fold_out'] = eval_out
        for params in grid:
            for eval_in in range(dict_param['k_inner']):
                dict_param['eval_fold_in'] = eval_in
                config = HyperparametersBaselines(dict_param)
                config.update(params)
                if os.path.exists(config.results_file()):
                    continue
                tr_data, ts_data = load_cv_baseline_data(config)
                estimator = config.params['model'](**config.get_suitable_args())
                estimator.fit(X=tr_data['ftr_in'], y=tr_data['score_in'])
                predictions = estimator.predict(X=ts_data['ftr_in'])
                test_loss = mean_squared_error(predictions, ts_data['score_in'])
                print('Test loss for the model %s is %.2f: ' % (config, test_loss))
                with open(config.results_file(), 'wb') as results_binary:
                    trait, = config.params['pers_traits_selection']
                    results = {'predictions': {trait: predictions},
                               'test_loss': {trait: test_loss},
                               'config': config}
                    pkl.dump(results, results_binary)


def baseline_eval_losses():
    inner_results = {}
    for result_file in os.listdir(baseline_result_dir):
        if result_file.startswith('predictions_'):
            # load the results for a suitable model into main memory
            with open(os.path.join(baseline_result_dir, result_file), 'rb') as out_result_file:
                results = pkl.load(out_result_file)
                config = results['config']
                model_name = config.get_name()
                outer_split = config.params['eval_fold_out']
                inner_split = config.params['eval_fold_in']
                trait, = config.params['pers_traits_selection']

                if outer_split not in inner_results.keys():
                    inner_results[outer_split] = {}
                if model_name not in inner_results[outer_split].keys():
                    inner_results[outer_split][model_name] = {}
                if inner_split not in inner_results[outer_split][model_name].keys():
                    inner_results[outer_split][model_name][inner_split] = {}
                # save the test losses for the particular inner split
                inner_results[outer_split][model_name][inner_split][trait] = results['test_loss'][trait]

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
    lin_reg = LinearRegression
    svr_reg = svm.SVR
    rvm_reg = RVR

    lin_reg_search = {'model': [lin_reg],
                      'fit_intercept': [True, False],
                      'normalize': [True, False]}
    svr_search = {'model': [svr_reg],
                  'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                  'epsilon': [0.1],
                  'gamma': [0.001],
                  'C': [1.0]}
    rvm_search = {'model': [rvm_reg],
                  'kernel': ['rbf', 'linear', 'poly'],
                  'n_iter': [500],
                  'alpha': [1e-06]}
    data_type = {'load_specific_data': [load_funct_data, load_struct_data],
                 'pers_traits_selection': [['NEO.NEOFAC_A'], ['NEO.NEOFAC_C'], ['NEO.NEOFAC_E'],
                                           ['NEO.NEOFAC_N'],
                                           ['NEO.NEOFAC_O']]}

    nested_cross_validation_baselines({**svr_search, **data_type})
    baseline_eval_losses()
