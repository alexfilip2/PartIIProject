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
    :return: void
    '''
    results = config.get_results()
    if results:
        return results
    else:
        print('Evaluating the GAT model %s...' % config)
    tr_data, ts_data = generate_splits_gat(gat_config=config, data=config.load_data())
    model = GATModel(config=config)
    model.fit(training_data=tr_data)
    return model.evaluate(test_data=ts_data)


def inner_nested_cv_gat():
    '''
     Performs the training/evaluation of GAT on the INNER CV LEVEL for each choice of outer split
    :return: dict of test losses and a lookup table to search the configuration of a model from its string name
    '''
    lookup_table = {}
    inner_results = {}
    grid = ParameterGrid(HyperparametersGAT.get_sampled_models())
    for eval_out in range(HyperparametersGAT().params['k_outer']):
        inner_results[eval_out] = {}
        for hyper_params in grid:
            # update the architecture hyper-parameters
            hyper_params['attention_heads'], hyper_params['hidden_units'] = hyper_params['arch_width']
            for eval_in in range(HyperparametersGAT().params['k_inner']):
                gat_model_config = HyperparametersGAT(hyper_params)
                gat_model_config.params['nested_CV_level'] = 'inner'
                gat_model_config.params['eval_fold_in'] = eval_in
                gat_model_config.params['eval_fold_out'] = eval_out
                # get base name of model without the NCV prefix
                model_name = gat_model_config.get_name()
                if model_name not in inner_results[eval_out].keys():
                    inner_results[eval_out][model_name] = {}
                # train and evaluate the GAT model
                inner_results[eval_out][model_name][eval_in] = evaluate_gat(gat_model_config)['test_loss']
                lookup_table[model_name] = gat_model_config

    return inner_results, lookup_table


def inner_losses_gat(filter_by_params: dict = {}):
    '''
     Retrieve the inner losses only of the GAT models using the values of the hyperparameters specified by the filter.
    :param filter_by_params: dict of hyperparameter name and value
    :return:
    '''
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


if __name__ == "__main__":
    evaluate_gat(HyperparametersGAT())
    inner_nested_cv_gat()
    pprint.pprint(inner_losses_gat({'learning_rate': 0.0005,
                                    'attn_drop': 0.4,
                                    'readout_aggregator': GATModel.master_node_aggregator}))
