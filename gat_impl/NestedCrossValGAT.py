from gat_impl.ExecuteGAT import *
from utils.LoadFunctionalData import *
from utils.LoadStructuralData import *
from sklearn.model_selection import ParameterGrid
from gat_impl.HyperparametersGAT import checkpts_dir


def sorted_stratification(config, data, unbalan_subj, lvl_split):
    saved_splt = 'split_%s_%d_%d_%d_%d_%s' % (lvl_split,
                                              config.params['k_outer'],
                                              config.params['k_inner'],
                                              config.params['eval_fold_out'],
                                              config.params['eval_fold_in'],
                                              config.params['pers_traits_selection'][0])

    saved_splt = os.path.join(config.proc_data_dir(), saved_splt + '.npy')
    if os.path.exists(saved_splt):
        print('Reload the split of sorted stratification for the model %s' % config)
        stratified_subj = np.load(saved_splt)

    else:
        from random import randint
        sorted_subjs_by_score = sorted(unbalan_subj, key=lambda x: data[x]['score_in'][0])
        k_split = config.params['k_outer'] if config.params['nested_CV_level'] == 'outer' else config.params[
            'k_inner']
        stratified_subj = []
        for _ in range(k_split):
            stratified_subj.append([])

        for window_nr in range(len(sorted_subjs_by_score) // k_split):
            window = sorted_subjs_by_score[window_nr * k_split:(window_nr + 1) * k_split]
            assert len(window) == k_split
            scores_left_window = k_split
            for fold in range(k_split):
                random_index_window = randint(0, scores_left_window - 1)
                stratified_subj[fold].append(window[random_index_window])
                del window[random_index_window]
                scores_left_window -= 1

        for rest in range(len(sorted_subjs_by_score) // k_split * k_split, len(sorted_subjs_by_score)):
            dump_fold_rest = randint(0, k_split - 1)
            stratified_subj[dump_fold_rest].append(sorted_subjs_by_score[rest])
        # save the particular split on disk
        stratified_subj = np.array(stratified_subj)
        np.save(saved_splt, stratified_subj)
    # retrieve the particular train/val/test sets of this split
    if config.params['nested_CV_level'] == 'outer':
        eval_fold = config.params['eval_fold_out']
    else:
        eval_fold = config.params['eval_fold_in']
    test_set = stratified_subj[eval_fold]
    # choose the fold before the test fold (circulary) as the validation one
    val_fold = eval_fold - 1
    val_set = stratified_subj[val_fold]
    stratified_subj = np.delete(np.delete(stratified_subj, obj=eval_fold, axis=0), obj=val_fold, axis=0)
    train_set = np.concatenate(stratified_subj)

    return train_set, val_set, test_set


def generate_cv_data(config, data, subjects):
    # prepare the outer split
    out_tr, out_vl, out_ts = sorted_stratification(unbalan_subj=subjects, data=data, config=config,
                                                   lvl_split='outer')
    if config.params['nested_CV_level'] == 'outer':
        tr_set = out_tr
        vl_set = out_vl
        ts_set = out_ts
        assert_disjoint([tr_set, vl_set, ts_set])
        return tr_set, vl_set, ts_set

    # prepare the inner split
    inner_data = np.concatenate([out_tr, out_vl])
    tr_set, vl_set, ts_set = sorted_stratification(unbalan_subj=inner_data, data=data, config=config, lvl_split='inner')
    assert_disjoint([tr_set, vl_set, ts_set])
    assert len(ts_set) < len(tr_set) and len(vl_set) < len(tr_set)
    return tr_set, vl_set, ts_set


def nested_cross_validation_gat():
    total_time_start = time.time()
    param_grid = {'hidden_units': [[5, 10, 10], [20, 20, 10], [30, 20, 10], [30, 40, 40]],
                  'attention_heads': [[3, 3, 2], [4, 4, 5], [2, 2, 3], [5, 7, 7]],
                  'learning_rate': [0.0001, 0.001],
                  'l2_coefficient': [0.0005],
                  'attn_drop': [0.6, 0.4, 0.2],
                  'include_ew': [True, False],
                  'readout_aggregator': [MainGAT.master_node_aggregator, MainGAT.concat_feature_aggregator,
                                         MainGAT.average_feature_aggregator
                                         ],
                  'load_specific_data': [load_struct_data, load_funct_data],
                  'pers_traits_selection': [['NEO.NEOFAC_A'], ['NEO.NEOFAC_O'], ['NEO.NEOFAC_C'], ['NEO.NEOFAC_N'],
                                            ['NEO.NEOFAC_E']],
                  'batch_size': [2, 4, 10]}

    grid = ParameterGrid(param_grid)
    dict_param = {
        'k_outer': 5,
        'k_inner': 5,
        'nested_CV_level': 'inner'}

    for eval_out in range(dict_param['k_outer']):
        dict_param['eval_fold_out'] = eval_out
        for params in grid:
            for eval_in in range(dict_param['k_inner']):
                dict_param['eval_fold_in'] = eval_in
                dict_param.update(params)
                config = HyperparametersGAT(dict_param)

                data, subjects = config.params['load_specific_data'](config.params)
                tr_set, vl_set, ts_set = generate_cv_data(config=config, data=data, subjects=subjects)
                model = CrossValidatedGAT(args=config)
                model.load_pipeline_data(data=data, train_subj=tr_set, val_subj=vl_set, test_subj=ts_set)
                model.build()
                model.train()
                model.test()
    cv_time = time.time() - total_time_start
    with open('time_elapsed.txt', 'w') as handle:
        handle.write(str(cv_time))


def extract_test_losses(param_search):
    model_descriptors = {}
    for file in os.listdir(checkpts_dir):
        if file.startswith('predictions_'):
            not_trained = False
            for param in param_search:
                if param not in file:
                    not_trained = True
            if not_trained:
                continue
            with open(os.path.join(checkpts_dir, file), 'rb') as out_result_file:
                results = pkl.load(out_result_file)

                name_model = file.split('_CV_')
                if name_model[0] not in model_descriptors.keys():
                    model_descriptors[name_model[0]] = {}
                model_descriptors[name_model[0]][name_model[1]] = results['test_loss']

    for model in model_descriptors.keys():
        if len(model_descriptors[model].values()) == 5:
            print('The test loss for model %s is %.3f' % (
                model, np.array(list(model_descriptors[model].values())).mean()))


if __name__ == "__main__":
    nested_cross_validation_gat()
