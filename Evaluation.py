from gat_impl.ConfigGAT import *
from baseline_impl.ConfigBaselines import *
from baseline_impl.InnerEvaluationBaselines import evaluate_baseline, inner_losses_baseline
from gat_impl.InnerEvaluationGAT import evaluate_gat, inner_losses_gat
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error


def get_best_models(model_name, data_set, trait) -> dict:
    '''
        Retrieve the best GAT/baseline models on the inner CV evaluated on a specific dataset and targeting a personality
     trait for each choice of outer evaluation
    :param model_name: str, required to differentiate between baselines
    :param data_set: specific loading function for the structural or functional data
    :param trait: personality trait targeted
    :return:
    '''
    # get the entire evaluation data for the inner CV of the filtered models
    if model_name == 'GAT':
        inner_results, lookup_table = inner_losses_gat(filter_by_params={'load_specific_data': data_set})
    else:
        inner_results, lookup_table = inner_losses_baseline(baseline_name=model_name,
                                                            filter_by_params={'load_specific_data': data_set})
    best_models = {}
    for out_split in inner_results.keys():
        best_models[out_split] = {}
        # average test loss on the inner splits when predicting a specific trait for each model
        avg_cv_inner_loss = {}
        for model in inner_results[out_split].keys():
            # accumulate each training loss for this trait for every inner split
            trait_inner_losses = []
            for in_split in inner_results[out_split][model].keys():
                if trait in inner_results[out_split][model][in_split].keys():
                    trait_inner_losses.append(inner_results[out_split][model][in_split][trait])
            assert len(trait_inner_losses) == ConfigGAT().params['k_inner']
            avg_cv_inner_loss[model] = np.mean(np.array(trait_inner_losses))
        else:
            IOError('The inner results for the outer evaluation choice %d does not exist' % out_split)

        sort_by_trait = list(sorted(avg_cv_inner_loss.keys(), key=lambda name: (avg_cv_inner_loss[name], name)))
        best_models[out_split] = (lookup_table[sort_by_trait[0]], avg_cv_inner_loss[sort_by_trait[0]])

    return best_models


def outer_evaluation(model_name, refresh=False):
    '''
    Outer evaluate the best inner models for a specific regression and return their outer MSE losses.
    :return: dictionary of the loss results over the outer folds for each choice of dataset and trait targeted
    '''
    outer_results_file = os.path.join(os.path.dirname((__file__)), 'Results', 'outer_evaluation_%s.pck' % model_name)
    if os.path.exists(outer_results_file) and not refresh:
        with open(outer_results_file, 'rb') as fp:
            return pkl.load(fp)

    outer_losses = {model_name: {}}
    # get only the hyper-parameter configurations that were inner evaluated
    for data_set in [load_struct_data, load_funct_data]:
        outer_losses[model_name][data_set] = {}
        out_eval_folds = list(range(ConfigGAT().params['k_outer']))
        traits = sorted(ConfigGAT().params['pers_traits_selection'])
        for trait in traits:
            outer_losses[model_name][data_set][trait] = {'loss': np.zeros(len(out_eval_folds)),
                                                         'best_models': []}
            best_specific_model = get_best_models(model_name=model_name, data_set=data_set, trait=trait)
            for eval_fold in out_eval_folds:
                config, _ = best_specific_model[eval_fold]
                # set the configuration object for the outer evaluation of the best inner model
                config.update({'nested_CV_level': 'outer',
                               'eval_fold_in': 0,
                               'eval_fold_out': eval_fold})
                # change the config for the particular trait only in case of baseline as GAT targets all at once
                if model_name != 'GAT':
                    config.params['pers_traits_selection'] = [trait]
                    config = ConfigBaselines(config.params)
                    evaluate_model = evaluate_baseline
                else:
                    evaluate_model = evaluate_gat
                    config = ConfigGAT(config.params)
                out_eval_loss = evaluate_model(config)['test_loss'][trait]
                outer_losses[model_name][data_set][trait]['loss'][eval_fold] = out_eval_loss
                outer_losses[model_name][data_set][trait]['best_models'].append(config)

    # save the outer evaluation losses
    with open(outer_results_file, 'wb') as handle:
        pkl.dump(outer_losses, handle)
    return outer_losses


def compute_metric(config, metric_name, trait):
    '''
    Compute the metric result on the outer evaluation for a model specified by a configuration object when predicting a
    specific trait
    :param config: configuration object
    :param metric_name: pearson-r or r-squared metric
    :param trait: personality trait targeted
    :return: value of the metric
    '''
    # if the model was not trained on outer CV, start its evaluation now
    results = evaluate_gat(config) if config.params['name'] == 'GAT' else evaluate_baseline(config)
    if metric_name == 'pearson_score':
        calculate_metric = pearsonr
    elif metric_name == 'r2_score':
        calculate_metric = r2_score
    elif metric_name == 'test_loss':
        calculate_metric = mean_squared_error
    else:
        raise ValueError('Possible metric:{pearson_score,r2_score}, not %s' % metric_name)
    observations, predictions = np.array(list(map(list, zip(*results['predictions'][trait]))))

    return calculate_metric(observations, predictions)


def get_outer_metrics(model_name):
    '''
     Prints the Pearson and R-squared metrics obtained from the outer predictions of the best-scoring GAT models per
     dataset and trait choices.
    :return: void
    '''
    outer_losses = outer_evaluation(model_name)[model_name]
    for data_set, data_set_dict in outer_losses.items():
        for trait, trait_dict in data_set_dict.items():
            pearson_values, pearson_p_values, r_squared_values, test_loss = [], [], [], []

            for eval_fold, best_config in enumerate(trait_dict['best_models']):
                pearson_tuple = compute_metric(best_config, 'pearson_score', trait)
                pearson_values.append(pearson_tuple[0])
                pearson_p_values.append(pearson_tuple[1])
                r_squared_values.append(compute_metric(best_config, 'r2_score', trait))
                test_loss.append(compute_metric(best_config, 'test_loss', trait))

            # average the metrics values over the outer folds
            avg_pearson = np.mean(np.array(pearson_values)), np.std(np.array(pearson_values))
            avg_p_value = np.mean(np.array(pearson_p_values))
            avg_r_squared = np.mean(np.array(r_squared_values)), np.std(np.array(r_squared_values))
            avg_test_loss = np.mean(np.array(test_loss)), np.std(np.array(test_loss))
            print('DATA: %s, predicting: %s' % (data_set.__name__, trait))
            print('(avg.PEARSON, STDEV): %s and p-VALUE %s: ' % (avg_pearson, avg_p_value))
            print('(avr.R-SQUARED, STDEV): %s ' % (avg_r_squared,))
            print('(avg. TEST LOSS, STDEV): %s \n' % (avg_test_loss,))
        print()


def validate_gat_inference():
    '''
     Apply a plain CV experiment on predicting a different target variable and report the pearson and r-squared.
    :return: void
    '''
    trait = 'FAC3'
    config = ConfigGAT({'pers_traits_selection': [trait],
                        'nested_CV_level': 'outer',
                        'eval_fold_in': 0,
                        'hidden_units': [30, 20, 15],
                        'attention_heads': [3, 3, 2],
                        'include_ew': False,
                        'readout_aggregator': GATModel.master_node_aggregator,
                        'load_specific_data': load_struct_data,
                        'learning_rate': 0.0001,
                        'attn_drop': 0.6})

    pearson_values, r_squared_values, p_values = [], [], []
    for out_fold in range(config.params['k_outer']):
        config.params['eval_fold_out'] = out_fold
        evaluate_gat(config)
        pearson_values.append(compute_metric(config, 'pearson_score', trait)[0])
        p_values.append(compute_metric(config, 'pearson_score', trait)[1])
        r_squared_values.append(compute_metric(config, 'r2_score', trait))

    # average the metric values over the outer folds
    avg_pearson = np.mean(np.array(pearson_values))
    avg_p_value = np.mean(np.array(p_values))
    avg_r_squared = np.mean(np.array(r_squared_values))
    print('Model %s | Pearson: %f | p-value: %s | R-squared: %f' % (config, avg_pearson, avg_p_value, avg_r_squared))


if __name__ == "__main__":
    outer_evaluation('GAT')
    outer_evaluation('LR')
    outer_evaluation('SVR')
    outer_evaluation('RVM')

