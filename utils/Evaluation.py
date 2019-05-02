from gat_impl.HyperparametersGAT import *
from baseline_impl.HyperparametersBaselines import *
from NestedCrossValGAT import load_cv_data
from baseline_impl.BaselineRegressions import evaluate_baseline
import numpy as np


def get_best_models(model_type, filter_by_params: dict) -> dict:
    '''
    Retrieve the best GAT/baseline models on the inner CV for each choice of outer evaluation set. Can also filter
    the models by specific hyper-parameters values they used as specified by filter_by_params.
    :param model_type: HyperparametersGAT or HyperparametersBaselines class to choose the model
    :param filter_by_params: dict of hyper-parameter name and its value
    :return: dict of the best models on each inner CV  per personality trait and their resulted evaluation loss
    '''
    # get the entire evaluation data for the inner CV of the filtered models
    inner_results, lookup_table = model_type.inner_losses(filter_by_params)
    best_losses = {}
    for out_split in list(range(model_type().params['k_outer'])):
        best_losses[out_split] = {}
        # average test loss on the inner splits when predicting a specific trait for each model
        avg_cv_inner_loss = {}
        if out_split in inner_results.keys():
            for model in inner_results[out_split].keys():
                avg_cv_inner_loss[model] = {}
                for trait in HyperparametersGAT().params['pers_traits_selection']:
                    # accumulate each training loss for this trait for every inner split
                    trait_inner_losses = []
                    for in_split in inner_results[out_split][model].keys():
                        if trait in inner_results[out_split][model][in_split]:
                            trait_inner_losses.append(inner_results[out_split][model][in_split][trait])
                    avg_cv_inner_loss[model][trait] = np.mean(np.array(trait_inner_losses))

        for trait in HyperparametersGAT().params['pers_traits_selection']:
            sort_by_trait = list(sorted(avg_cv_inner_loss.keys(), key=lambda name: avg_cv_inner_loss[name][trait]))
            best_losses[out_split][trait] = (lookup_table[sort_by_trait[0]], avg_cv_inner_loss[sort_by_trait[0]][trait])
    return best_losses


def bootstrap_gat():
    '''
    Run the best GAT model obtained from the inner CV for each outer CV multiple times in order to estimate its
    prediction performance in terms of the mean and standard deviation of the evaluation loss resulted
    :return: dict of a list of evaluation losses obtained for the best models on a specific dataset and trait choice
    '''
    BOOTSTRAP_FREQUENCY = 50
    outer_losses = {'GAT': {}}
    bootstrap_file = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                                  'bootstrap_gat.pck')
    if os.path.exists(bootstrap_file):
        with open(bootstrap_file, 'rb') as fp:
            outer_losses = pkl.load(fp)
    sampled_hyper = HyperparametersGAT.get_sampled_models()
    for data_set in sampled_hyper['load_specific_data']:
        if data_set not in outer_losses.keys():
            outer_losses[data_set] = {}
        best_gat = get_best_models(model_type=HyperparametersGAT, filter_by_params={'load_specific_data': data_set})
        out_eval_folds = sorted(list(range(HyperparametersBaselines().params['k_outer'])))
        traits = sorted(HyperparametersGAT().params['pers_traits_selection'])
        for eval_fold in out_eval_folds:
            if eval_fold not in outer_losses[data_set].keys():
                outer_losses[data_set][eval_fold] = {}
            for trait in traits:
                if trait not in outer_losses[data_set][eval_fold].keys():
                    outer_losses[data_set][eval_fold][trait] = []
                else:
                    # either all the BOOTSTRAP_FREQUENCY models were evaluated and saved on disk or none
                    continue
                if eval_fold in best_gat.keys():
                    if trait in best_gat[eval_fold].keys():
                        gat_config, _ = best_gat[eval_fold][trait]
                        # set the configuration object for the outer evaluation of the best inner model
                        gat_config.params['nested_CV_level'] = 'outer'
                        tr_data, vl_data, ts_data = load_cv_data(gat_config=gat_config)
                        for _ in range(BOOTSTRAP_FREQUENCY):
                            model = GATModel(config=gat_config)
                            model.fit(training_data=tr_data, validation_data=vl_data)
                            loss_result = model.evaluate(test_data=ts_data)['test_loss'][trait]
                            outer_losses[data_set][eval_fold][trait].append(loss_result)
                        with open(bootstrap_file, 'wb') as fp:
                            pkl.dump(outer_losses, fp, protocol=pkl.HIGHEST_PROTOCOL)
    return outer_losses


def outer_evaluation_gat():
    '''
    Retrieve the outer CV results of GAT from the bootstrapping
    :return: the mean loss and std dev of the multiple evaluation losses per dataset and personality trait
    '''
    outer_losses = bootstrap_gat()
    mean_out_losses = outer_losses.copy()
    for data_set, data_set_data in outer_losses['GAT'].items():
        print('For the % dataset' % str(data_set.__name__.split('_')[1]))
        for eval_fold, eval_fold_data in data_set_data.items():
            print('\t For the outer evaluation fold %d' % eval_fold)
            for trait, bootstrap__pool in eval_fold_data.items():
                stdev = np.std(np.array(bootstrap__pool))
                mean = np.mean(np.array(bootstrap__pool))
                mean_out_losses[data_set][eval_fold][trait] = mean
                print('\t\t For the trait %s the mean loss is %f and the stdev is %f ' % (trait, mean, stdev))
            print()
        print()
    return mean_out_losses


def outer_evaluation_baselines():
    '''
     Outer evaluates best baseline model for each choice of regression model, dataset and personality trait
    :return: dictionary of the evaluation resutls of these models on each outer evaluation fold
    '''
    outer_losses = {}
    outer_results_file = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                                      'baseline_outer_eval_losses.pck')
    if os.path.exists(outer_results_file):
        with open(outer_results_file, 'rb') as fp:
            outer_losses = pkl.load(fp)
            return outer_losses
    # decalre the baseline models ID's
    baselines = ['RVM', 'LR', 'SVR']
    for baseline_name in baselines:
        if baseline_name not in outer_losses.keys():
            outer_losses[baseline_name] = {}
        # get only the hyper-parameter configurations that were inner evaluated
        sampled_hyper = HyperparametersBaselines.get_sampled_models(baseline_name)
        for data_set in sampled_hyper['load_specific_data']:
            if data_set not in outer_losses[baseline_name].keys():
                outer_losses[baseline_name][data_set] = {}
            best_specific_baseline = get_best_models(model_type=HyperparametersBaselines,
                                                     filter_by_params={'load_specific_data': data_set,
                                                                       'name': baseline_name})
            out_eval_folds = sorted(list(range(HyperparametersBaselines().params['k_outer'])))
            traits = sorted(HyperparametersGAT().params['pers_traits_selection'])
            for eval_fold in out_eval_folds:
                if eval_fold not in outer_losses[baseline_name][data_set].keys():
                    outer_losses[baseline_name][data_set][eval_fold] = {}
                for trait in traits:
                    if eval_fold in best_specific_baseline.keys():
                        if trait in best_specific_baseline[eval_fold].keys():
                            baseline_config, _ = best_specific_baseline[eval_fold][trait]
                            # set the configuration object for the outer evaluation of the best inner model
                            baseline_config.params['nested_CV_level'] = 'outer'
                            outer_losses[baseline_name][data_set][eval_fold][trait] = evaluate_baseline(baseline_config)

    with open(outer_results_file, 'wb') as outer_losses_binary:
        pkl.dump(outer_losses, outer_losses_binary, protocol=pkl.HIGHEST_PROTOCOL)
    return outer_losses


if __name__ == "__main__":
    bootstrap_gat()
