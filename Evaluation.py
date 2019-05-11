from gat_impl.HyperparametersGAT import *
from baseline_impl.HyperparametersBaselines import *
from baseline_impl.NestedCrossValBaselines import evaluate_baseline, inner_losses_baseline
from NestedCrossValGAT import evaluate_gat, inner_losses_gat
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import pprint


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
            assert len(trait_inner_losses) == HyperparametersGAT().params['k_inner']
            avg_cv_inner_loss[model] = np.mean(np.array(trait_inner_losses))
        else:
            IOError('The inner results for the outer evaluation choice %d does not exist' % out_split)

        sort_by_trait = list(sorted(avg_cv_inner_loss.keys(), key=lambda name: (avg_cv_inner_loss[name], name)))
        best_models[out_split] = (lookup_table[sort_by_trait[0]], avg_cv_inner_loss[sort_by_trait[0]])

    return best_models


def outer_evaluation(model_name, refresh=False):
    '''
    Run the best GAT model obtained from the inner CV for each outer CV multiple times in order to estimate its
    prediction performance in terms of the mean and standard deviation of the evaluation loss resulted
    :return: dict of a list of evaluation losses obtained for the best models on a specific dataset and trait choice
    '''
    outer_results_file = os.path.join(os.path.dirname((__file__)), 'Results', 'outer_evaluation_%s.pck' % model_name)
    if os.path.exists(outer_results_file) and not refresh:
        with open(outer_results_file, 'rb') as fp:
            return pkl.load(fp)

    outer_losses = {model_name: {}}
    # get only the hyper-parameter configurations that were inner evaluated
    for data_set in [load_struct_data, load_funct_data]:
        outer_losses[model_name][data_set] = {}
        out_eval_folds = list(range(HyperparametersGAT().params['k_outer']))
        traits = sorted(HyperparametersGAT().params['pers_traits_selection'])
        for trait in traits:
            outer_losses[model_name][data_set][trait] = {'loss': np.zeros(len(out_eval_folds)),
                                                         'best_models': []}
            best_specific_model = get_best_models(model_name=model_name, data_set=data_set, trait=trait)
            for eval_fold in out_eval_folds:
                config, _ = best_specific_model[eval_fold]
                # set the configuration object for the outer evaluation of the best inner model
                config.params['nested_CV_level'] = 'outer'
                config.params['eval_fold_in'] = 0
                config.params['eval_fold_out'] = eval_fold
                # change the config for the particular trait only in case of baseline as GAT targets all at once
                if model_name != 'GAT':
                    config.params['pers_traits_selection'] = [trait]
                    evaluate_model = evaluate_baseline
                else:
                    evaluate_model = evaluate_gat
                out_eval_loss = evaluate_model(config)['test_loss'][trait]
                outer_losses[model_name][data_set][trait]['loss'][eval_fold] = out_eval_loss
                outer_losses[model_name][data_set][trait]['best_models'].append(config)

    # save the outer evaluation losses
    with open(outer_results_file, 'wb') as handle:
        pkl.dump(outer_losses, handle)

    return outer_losses


def compute_metric(config, metric_name, trait):
    # if the model was not trained on outer CV, start its evaluation now
    if config.params['name'] == 'GAT':
        results = evaluate_gat(config)
    else:
        results = evaluate_baseline(config)
    if metric_name == 'pearson_score':
        calculate_metric = pearsonr
    elif metric_name == 'r2_score':
        calculate_metric = r2_score
    else:
        raise ValueError('Possible metric:{pearson_score,r2_score}, not %s' % metric_name)
    observations, predictions = zip(*results['predictions'][trait])
    observations, predictions = np.array(list(observations)), np.array(list(predictions))

    return calculate_metric(observations, predictions)


def get_outer_metrics(model_name):
    '''
     Prints the Pearson and R-squared metrics obtained from the outer predictions of the best-scoring GAT models per
     dataset and trait choices.
    :return: void
    '''
    outer_losses = outer_evaluation(model_name)[model_name]
    for data_set, dataset_dict in outer_losses.items():
        for trait, trait_dict in dataset_dict.items():
            pearson = []
            pearson_p_value = []
            r_squared = []
            for eval_fold, best_config in enumerate(trait_dict['best_models']):
                pearson_tuple = compute_metric(best_config, 'pearson_score', trait)
                pearson.append(pearson_tuple[0])
                pearson_p_value.append(pearson_tuple[1])
                r_squared.append(compute_metric(best_config, 'r2_score', trait))

            # average the metrics values over the outer folds
            pearson_value = np.mean(np.array(pearson)), np.std(np.array(pearson))
            r_squared_value = np.mean(np.array(r_squared)), np.std(np.array(r_squared))
            pearson_p_value = np.mean(np.array(pearson_p_value))
            print(
                'Dataset: %s, predicting: %s, (avg. PEARSON, STDEV): %s and P-VALUE %s and (avr. R-SQUARED, STDEV): %s' % (
                    data_set.__name__, trait, pearson_value, pearson_p_value, r_squared_value))


def validate_gat_inference():
    trait = 'FAC3'
    gat_config = HyperparametersGAT({'pers_traits_selection': [trait]}
                                    )
    rvm_config = HyperparametersBaselines({'name': 'RVM',
                                           'kernel': 'linear',
                                           'n_iter': 100,
                                           'pers_traits_selection': [trait],
                                           'alpha': 1e-06})
    ridge_config = HyperparametersBaselines({'name': 'LR',
                                             'pers_traits_selection': [trait],
                                             'alpha': 10})
    svr_config = HyperparametersBaselines({'name': 'SVR',
                                           'kernel': 'rbf',
                                           'pers_traits_selection': [trait],
                                           'gamma': 0.5,
                                           'C': 1})
    models = [gat_config, rvm_config, ridge_config, svr_config]
    for config in models:
        config.params['nested_CV_level'] = 'outer'
        pearson, r_squared = [], []
        for out_fold in range(gat_config.params['k_outer']):
            config.params['eval_fold_out'] = out_fold
            if config.params['name'] == 'GAT':
                evaluate_gat(config)
            else:
                evaluate_baseline(config)
            pearson.append(compute_metric(config, 'pearson_score', trait)[0])
            r_squared.append(compute_metric(config, 'r2_score', trait))
        pearson_value = np.mean(np.array(pearson))
        r_squared_value = np.mean(np.array(r_squared))
        print('For model %s, Pearson: %f and R-squared: %f' % (config.params['name'], pearson_value, r_squared_value))


if __name__ == "__main__":
    get_outer_metrics(model_name='GAT')
