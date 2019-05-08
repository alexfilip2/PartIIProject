from gat_impl.HyperparametersGAT import *
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
import pickle as pkl
import os
import pprint


def generate_splits_gat(gat_config: HyperparametersGAT, data: dict, outer_flag=True) -> tuple:
    '''
     Splits the entire dataset provided for the Nested CV as specified by the GAT configuration object
    :param gat_config: configuration object
    :param data: the entire formatted dataset
    :param outer_flag: the recursive function behaves differently when splitting for outer level NCV
    :return: training and evaluation datasets
    '''
    outer_tr_data, outer_ts_data = {}, {}
    if outer_flag:
        k_split = gat_config.params['k_outer']
        eval_fold = gat_config.params['eval_fold_out']
    else:
        k_split = gat_config.params['k_inner']
        eval_fold = gat_config.params['eval_fold_in']

    for input_type in data.keys():
        splitter_out = KFold(k_split)
        train_indices, test_indices = list(splitter_out.split(X=data[input_type]))[eval_fold]
        outer_tr_data[input_type] = data[input_type][train_indices]
        outer_ts_data[input_type] = data[input_type][test_indices]

    if gat_config.params['nested_CV_level'] == 'inner' and outer_flag:
        inner_tr_data, inner_ts_data = generate_splits_gat(gat_config, outer_tr_data, outer_flag=False)
        return inner_tr_data, inner_ts_data
    else:
        return outer_tr_data, outer_ts_data


def evaluate_gat(config: HyperparametersGAT) -> dict:
    '''
    Train and evaluate the GAT model specified by the GAT configuration object
    :param config: configuration object for the GAT model
    :return:
    '''
    results = config.get_results()
    if results:
        return results
    tr_data, ts_data = generate_splits_gat(gat_config=config, data=config.load_data())
    model = GATModel(config=config)
    model.fit(training_data=tr_data)
    return model.evaluate(test_data=ts_data)


# perform the Nested Cross Validation
def inner_nested_cv_gat():
    lookup_table = {}
    inner_results = {}
    grid = ParameterGrid(HyperparametersGAT.get_sampled_models())
    ncv_params = {'k_outer': HyperparametersGAT().params['k_outer'],
                  'k_inner': HyperparametersGAT().params['k_inner'],
                  'nested_CV_level': 'inner'}
    for params in grid:
        # update the architecture hyperparameters
        params['attention_heads'] = params['arch_width'][0]
        params['hidden_units'] = params['arch_width'][1]
        gat_model_config = HyperparametersGAT(params)
        for eval_out in range(ncv_params['k_outer']):
            inner_results[eval_out] = {}
            ncv_params['eval_fold_out'] = eval_out
            for eval_in in range(ncv_params['k_inner']):
                ncv_params['eval_fold_in'] = eval_in
                # update the nested CV hyperparameters
                gat_model_config.update(ncv_params)
                model_name = gat_model_config.get_name()
                if model_name not in inner_results[eval_out].keys():
                    inner_results[eval_out][model_name] = {}
                # train and evaluate the GAT model
                inner_results[eval_out][model_name][eval_in] = evaluate_gat(gat_model_config)['test_loss']
                lookup_table[model_name] = gat_model_config

    return inner_results, lookup_table


def inner_losses_gat(filter_by_params: dict = {}):
    inner_losses_file = os.path.join(os.path.dirname(__file__), 'Results', 'gat_inner_eval_losses.pck')
    if os.path.exists(inner_losses_file):
        with open(inner_losses_file, 'rb') as fp:
            inner_results, lookup_table = pkl.load(fp)
    else:
        inner_results, lookup_table = inner_nested_cv_gat()
        with open(inner_losses_file, 'wb') as handle:
            pkl.dump((inner_results, lookup_table), handle)
    # extract only the evaluation results of the models with specific hyper-parameters
    for out_split in inner_results.keys():
        model_names = list(inner_results[out_split].keys())
        for model in model_names:
            if not filter_by_params.items() <= lookup_table[model].params.items():
                inner_results[out_split].pop(model)

    return inner_results, lookup_table


def extract_partial_results():
    inner_models_dir = os.path.join(os.path.dirname(__file__), 'Results', 'GAT_results')
    for file in os.listdir(inner_models_dir):
        if file.startswith('predictions'):
            with open(os.path.join(inner_models_dir, file), 'rb') as fp:
                results = pkl.load(fp)
                print(HyperparametersGAT(results['params']))
                pprint.pprint(results['test_loss'])


if __name__ == "__main__":
    extract_partial_results()
