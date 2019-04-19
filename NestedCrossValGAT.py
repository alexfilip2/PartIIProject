from gat_impl.ExecuteGAT import *
from utils.LoadFunctionalData import *
from utils.LoadStructuralData import *
from sklearn.model_selection import ParameterGrid
from gat_impl.HyperparametersGAT import checkpts_dir
import itertools
import math


def sorted_stratification(config, data, unbalan_subj, lvl_split):
    split_id = 'split_%s_%d_%d_%d_%d_%s' % (lvl_split,
                                            config.params['k_outer'],
                                            config.params['k_inner'],
                                            config.params['eval_fold_out'],
                                            config.params['eval_fold_in'],
                                            ''.join(config.params['pers_traits_selection']).replace('NEO.NEOFAC_', ''))

    split_path = os.path.join(config.proc_data_dir(), split_id + '.pck')
    # identify the particular folds determined by the configuration of the model
    if config.params['nested_CV_level'] == 'outer':
        k_split = config.params['k_outer']
        eval_fold = config.params['eval_fold_out']
    else:
        k_split = config.params['k_inner']
        eval_fold = config.params['eval_fold_in']

    # check if the particular split already saved on disk
    if os.path.exists(split_path):
        print('Reload the split of sorted stratification for the model %s' % config)
        with open(split_path, 'rb') as split_binary:
            stratified_subj = pickle.load(split_binary)
    else:
        from random import randint
        sorted_subjects = sorted(unbalan_subj, key=lambda subj_name: sum(data[subj_name]['score_in']))
        stratified_subj = [[] for _ in range(k_split)]

        for window_nr in range(len(sorted_subjects) // k_split):
            window = sorted_subjects[window_nr * k_split:(window_nr + 1) * k_split]
            assert len(window) == k_split
            for fold in range(k_split):
                random_index_window = randint(0, len(window) - 1)
                stratified_subj[fold].append(window[random_index_window])
                del window[random_index_window]

        # dump the rest of examples uniformly at random to the folds constructed so far
        for unassigned_elem in range(len(sorted_subjects) // k_split * k_split, len(sorted_subjects)):
            dump_fold_id = randint(0, k_split - 1)
            stratified_subj[dump_fold_id].append(sorted_subjects[unassigned_elem])
        # save the particular split on disk
        with open(split_path, 'wb') as split_binary:
            pickle.dump(stratified_subj, split_binary, protocol=pkl.HIGHEST_PROTOCOL)

    # retrieve the particular train/val/test sets of this split
    test_set = stratified_subj.pop(eval_fold)
    # choose the fold before the test fold (circularly) as the validation one
    val_fold = eval_fold - 1
    val_set = stratified_subj.pop(val_fold)
    # delete the val/test sets from the entire data
    train_set = list(itertools.chain.from_iterable(stratified_subj))

    return train_set, val_set, test_set


def generate_cv_data(config, data, subjects):
    # prepare the outer split
    tr_set, vl_set, ts_set = sorted_stratification(unbalan_subj=subjects, data=data, config=config, lvl_split='outer')
    if config.params['nested_CV_level'] == 'outer':
        assert_disjoint([tr_set, vl_set, ts_set])
        return tr_set, vl_set, ts_set

    # prepare the inner split
    inner_data = list(itertools.chain.from_iterable([tr_set, vl_set]))
    tr_set, vl_set, ts_set = sorted_stratification(unbalan_subj=inner_data, data=data, config=config, lvl_split='inner')
    assert_disjoint([tr_set, vl_set, ts_set])
    assert len(ts_set) < len(tr_set) and len(vl_set) < len(tr_set)
    return tr_set, vl_set, ts_set


NO_LAYERS = 3


def sample_hyper_params(max_samples=10080):
    sampled_models_file = os.path.join(os.path.join(os.path.dirname(__file__)), 'sampled_models.pck')
    if os.path.exists(sampled_models_file):
        with open(sampled_models_file, 'rb') as handle:
            param_grid = pickle.load(handle)
            return ParameterGrid(param_grid)
    param_grid = {
        'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'decay_rate': [0.0005],
        'attn_drop': [0.0, 0.2, 0.4, 0.6, 0.8],
        'readout_aggregator': [GATModel.average_feature_aggregator, GATModel.master_node_aggregator,
                               GATModel.concat_feature_aggregator],
        'load_specific_data': [load_struct_data, load_funct_data],
        'include_ew': [True, False],
        'batch_size': [32]}
    models_so_far = np.prod(np.array([len(param_grid[x]) for x in param_grid.keys()])) / 25
    sampling_count = math.floor(max_samples / models_so_far)
    sample_ah = list(itertools.product(range(3, 7), repeat=3))
    sample_hu = list(itertools.product(range(12, 48), repeat=3))

    def filter_features_expansion(ah_hu_choice):
        for i in range(1, len(ah_hu_choice[0]) - 1):
            if ah_hu_choice[0][i] * ah_hu_choice[1][i] > ah_hu_choice[0][i - 1] * ah_hu_choice[1][i - 1]:
                return False
        if ah_hu_choice[1][i + 1] > ah_hu_choice[0][i] * ah_hu_choice[1][i]:
            return False
        return True

    all_choices = set(
        filter(lambda x: filter_features_expansion(x), list(itertools.product(sample_ah, sample_hu))))
    param_grid['arch_width'] = list(map(lambda x: [list(x[0]), list(x[1])], random.sample(all_choices, sampling_count)))
    with open(sampled_models_file, 'wb') as handle:
        pickle.dump(param_grid, handle)

    return ParameterGrid(param_grid)


# perform the Nested Cross Validation
def nested_cross_validation_gat():
    grid = sample_hyper_params()
    dict_param = {
        'k_outer': 5,
        'k_inner': 5,
        'nested_CV_level': 'inner'}
    for eval_out in range(dict_param['k_outer']):
        dict_param['eval_fold_out'] = eval_out
        for params in grid:
            params['attention_heads'] = params['arch_width'][0]
            params['hidden_units'] = params['arch_width'][1]
            for eval_in in range(dict_param['k_inner']):
                dict_param['eval_fold_in'] = eval_in
                dict_param.update(params)
                config = HyperparametersGAT(dict_param)
                if os.path.exists(config.results_file()):
                    continue
                data, subjects = config.params['load_specific_data'](config.params)
                tr_set, vl_set, ts_set = generate_cv_data(config=config, data=data, subjects=subjects)
                model = GATModel(args=config)
                model.fit(data=data, train_subj=tr_set, val_subj=vl_set)
                model.test(data=data, test_subj=ts_set)


def extract_test_losses():
    inner_results = {}
    for result_file in os.listdir(checkpts_dir):
        if result_file.startswith('predictions_'):
            # load the results for a suitable model into main memory
            with open(os.path.join(checkpts_dir, result_file), 'rb') as out_result_file:
                results = pkl.load(out_result_file)
                model_name, cv_details = result_file.split('_CV_')
                outer_split = cv_details[1]
                inner_split = cv_details[0]

                if outer_split not in inner_results.keys():
                    inner_results[outer_split] = {}
                if model_name not in inner_results[outer_split].keys():
                    inner_results[outer_split][model_name] = {}
                # save the test losses for the particular inner split
                inner_results[outer_split][model_name][inner_split] = results['test_loss']

    # print out the best average test loss (as a progress ?/nr of splits)
    best_losses = {}
    for out_split in inner_results.keys():
        best_losses[out_split] = {}
        avg_loss = {}
        print('The results for the outer split of ID %s are: ' % out_split)
        print()
        for model in inner_results[out_split].keys():
            split_losses = list(inner_results[out_split][model].values())
            if len(split_losses) == 5:
                avg_loss[model] = {}
                for trait_key in sorted(split_losses[0].keys()):
                    avg_loss[model][trait_key] = sum([d[trait_key] for d in split_losses]) / len(split_losses)

        for trait_key in sorted(avg_loss[list(avg_loss.keys())[0]]):
            sort_by_trait = list(sorted(avg_loss.keys(), key=lambda key: avg_loss[key][trait_key]))
            best_losses[out_split][trait_key] = (sort_by_trait[0], avg_loss[sort_by_trait[0]][trait_key])
            print(
                'The BEST average test loss for trait %s is  %.3f' % (trait_key, avg_loss[sort_by_trait[0]][trait_key]))
            print('The model achieving this score for %s is %s' % (trait_key, sort_by_trait[0]))
            for trait, loss in avg_loss[sort_by_trait[0]].items():
                print(trait, loss)


# perform the Nested Cross Validation
def test_ncv():
    param_grid = {'hidden_units': [[20, 20, 10]],
                  'attention_heads': [[3, 3, 2]],
                  'readout_aggregator': [GATModel.average_feature_aggregator],
                  'load_specific_data': [load_struct_data, load_funct_data],
                  'include_ew': [True, False],
                  'attn_drop': [0.4],
                  'learning_rate': [0.0001],
                  'decay_rate': [0.0005],
                  'batch_size': [8]}

    # perform the Nested Cross Validation
    grid = ParameterGrid(param_grid)
    dict_param = {
        'k_outer': 5,
        'k_inner': 5,
        'nested_CV_level': 'inner'}
    for eval_out in range(1):
        dict_param['eval_fold_out'] = eval_out
        for params in grid:
            for eval_in in range(dict_param['k_inner']):
                dict_param['eval_fold_in'] = eval_in
                dict_param.update(params)
                config = HyperparametersGAT(dict_param)
                if os.path.exists(config.checkpt_file()):
                    continue
                data, subjects = config.params['load_specific_data'](config.params)
                tr_set, vl_set, ts_set = generate_cv_data(config=config, data=data, subjects=subjects)
                model = GATModel(args=config)
                model.fit(data=data, train_subj=tr_set, val_subj=vl_set)
                model.test(data=data, test_subj=ts_set)


if __name__ == "__main__":
    extract_test_losses()
