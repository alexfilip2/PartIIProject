from gat_impl.HyperparametersGAT import *
from baseline_impl.HyperparametersBaselines import *
from baseline_impl.BaselineRegressions import evaluate_baseline
from NestedCrossValGAT import evaluate_gat
import numpy as np
import pprint


def get_best_models(model_type, model_name, data_set, trait) -> dict:
    '''
        Retrieve the best GAT/baseline models on the inner CV evaluated on a specific dataset and targeting a personality
     trait for each choice of outer evaluation
    :param model_type: HyperparametersGAT or HyperparametersBaselines class to choose the model
    :param model_name: str, required to differentiate between baselines
    :param data_set: specific loading function for the structural or functional data
    :param trait: personality trait targeted
    :return:
    '''
    # get the entire evaluation data for the inner CV of the filtered models
    inner_results, lookup_table = model_type.inner_losses(filter_by_params={'load_specific_data': data_set,
                                                                            'name': model_name})
    best_losses = {}
    for out_split in list(range(model_type().params['k_outer'])):
        best_losses[out_split] = {}
        # average test loss on the inner splits when predicting a specific trait for each model
        avg_cv_inner_loss = {}
        if out_split in inner_results.keys():
            for model in inner_results[out_split].keys():
                # accumulate each training loss for this trait for every inner split
                trait_inner_losses = []
                for in_split in inner_results[out_split][model].keys():
                    if trait in inner_results[out_split][model][in_split].keys():
                        trait_inner_losses.append(inner_results[out_split][model][in_split][trait])
                assert len(trait_inner_losses) == 5
                avg_cv_inner_loss[model] = np.mean(np.array(trait_inner_losses))
        else:
            IOError('The inner results for the outer evaluation choice %d does not exist' % out_split)
        sort_by_trait = list(sorted(avg_cv_inner_loss.keys(), key=lambda name: (avg_cv_inner_loss[name], name)))
        best_losses[out_split] = (lookup_table[sort_by_trait[0]], avg_cv_inner_loss[sort_by_trait[0]])
    return best_losses


def outer_evaluation():
    '''
    Run the best GAT model obtained from the inner CV for each outer CV multiple times in order to estimate its
    prediction performance in terms of the mean and standard deviation of the evaluation loss resulted
    :return: dict of a list of evaluation losses obtained for the best models on a specific dataset and trait choice
    '''
    outer_results_file = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                                      'outer_evaluation.pck')
    if os.path.exists(outer_results_file):
        with open(outer_results_file, 'rb') as fp:
            outer_losses = pkl.load(fp)
            pprint.pprint(outer_losses)
            return outer_losses

    outer_losses = {}
    # declare the models ID's
    models = ['RVM', 'LR', 'SVR', 'GAT']
    for model_name in models:
        outer_losses[model_name] = {}
        model_class = HyperparametersGAT if model_name == 'GAT' else HyperparametersBaselines
        # get only the hyper-parameter configurations that were inner evaluated
        sampled_hyper = model_class.get_sampled_models(model_name)
        for data_set in sampled_hyper['load_specific_data']:
            outer_losses[model_name][data_set] = {}
            out_eval_folds = sorted(list(range(model_class().params['k_outer'])))
            traits = sorted(HyperparametersGAT().params['pers_traits_selection'])
            for eval_fold in out_eval_folds:
                outer_losses[model_name][data_set][eval_fold] = {}
                for trait in traits:
                    best_specific_model = get_best_models(model_type=model_class,
                                                          model_name=model_name,
                                                          data_set=data_set,
                                                          trait=trait)
                    config, _ = best_specific_model[eval_fold]
                    # set the configuration object for the outer evaluation of the best inner model
                    config.params['nested_CV_level'] = 'outer'
                    config.params['eval_fold_out'] = eval_fold
                    # change the config for the particular trait only in case of baseline as GAT targets all at once
                    if model_name != 'GAT':
                        config.params['pers_traits_selection'] = [trait]
                        evaluate_model = evaluate_baseline
                    else:
                        evaluate_model = evaluate_gat
                    outer_losses[model_name][data_set][eval_fold][trait] = evaluate_model(config)['test_loss'][trait]

    # save the outer evaluation losses
    with open(outer_results_file, 'wb') as handle:
        pkl.dump(outer_losses, handle)
    pprint.pprint(outer_losses)
    return outer_losses


def get_outer_metrics():
    '''
     Prints the Pearson and R-squared metrics obtained from the outer predictions of the best-scoring GAT models per
     dataset and trait choices.
    :return: void
    '''
    for data_set in [load_funct_data, load_struct_data]:
        traits = sorted(HyperparametersGAT().params['pers_traits_selection'])
        for trait in traits:
            best_gat = get_best_models(HyperparametersGAT, model_name='GAT', data_set=data_set, trait=trait)
            pearson = []
            r_squared = []
            for out_fold in range(HyperparametersGAT().params['k_outer']):
                config, _ = best_gat[out_fold]
                config.params['nested_CV_level'] = 'outer'
                config.params['eval_fold_out'] = out_fold
                results = config.get_results()
                pearson.append(results['pearson'][trait][0])
                r_squared.append(results['r2_score'][trait])
            pearson_value = (np.mean(np.array(pearson)), np.std(np.array(pearson)))
            r_squared_value = (np.mean(np.array(r_squared)), np.std(np.array(r_squared)))

            print('On the dataset %s when predicting %s, the avg. pearson value was %s and the r-squared value %s' % (
                data_set.__name__.split('_')[1], trait, pearson_value, r_squared_value))


if __name__ == "__main__":
    get_outer_metrics()
