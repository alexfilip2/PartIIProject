from gat_impl.ExecuteGAT import *
from utils.LoadFunctionalData import *
from utils.LoadStructuralData import *
from sklearn.model_selection import ParameterGrid
from gat_impl.HyperparametersGAT import checkpts_dir
import itertools


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


def nested_cross_validation_gat():
    param_grid = {'hidden_units': [[21, 20]],
                  'attention_heads': [[6, 7]],
                  'readout_aggregator': [GATModel.average_feature_aggregator,
                                         GATModel.concat_feature_aggregator],
                  'load_specific_data': [load_struct_data, load_funct_data],
                  'include_ew': [True],
                  'attn_drop': [0.4],
                  'learning_rate': [0.0001],
                  'decay_rate': [0.0005],
                  'batch_size': [2]}

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
                data, subjects = config.params['load_specific_data'](config.params)
                tr_set, vl_set, ts_set = generate_cv_data(config=config, data=data, subjects=subjects)
                model = GATModel(args=config)
                model.fit(data=data, train_subj=tr_set, val_subj=vl_set)
                model.test(data=data, test_subj=ts_set)


def extract_test_losses(param_search=[]):
    model_descriptors = {}
    for result_file in os.listdir(checkpts_dir):
        if result_file.startswith('predictions_'):
            # filter out the models not using other parameters than param_search
            not_trained = False
            for param in param_search:
                if param not in result_file:
                    not_trained = True
            if not_trained:
                continue
            # load the results for a suitable model into main memory
            with open(os.path.join(checkpts_dir, result_file), 'rb') as out_result_file:
                results = pkl.load(out_result_file)
                model_name, cv_details = result_file.split('_CV_')
                model_name = model_name + '_outer_split_' + cv_details[1]
                if model_name not in model_descriptors.keys():
                    model_descriptors[model_name] = {}
                # save the test losses for the particular inner split
                model_descriptors[model_name][cv_details[0]] = results['test_loss']

    # print out the average test loss (as a progress ?/nr of splits)
    for model in model_descriptors.keys():
        print('The test losses for model  %s are: ' % model)
        split_losses = list(model_descriptors[model].values())
        avg_loss = {}
        for trait_key in sorted(split_losses[0].keys()):
            avg_loss[trait_key] = sum([d[trait_key] for d in split_losses]) / len(split_losses)
            print('The average test loss for trait %s is %.3f calculated yet on %d/5 folds' % (
                trait_key, avg_loss[trait_key], len(split_losses)))


if __name__ == "__main__":
    nested_cross_validation_gat()
