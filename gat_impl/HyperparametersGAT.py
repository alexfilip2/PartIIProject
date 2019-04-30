from utils.LoadStructuralData import load_struct_data, dir_structural_data
from utils.LoadFunctionalData import load_funct_data, dir_functional_data
from utils.ToolsDataProcessing import lower_bound_filter
from keras.activations import relu
import pickle as pkl
from gat_impl.ExecuteGAT import *
import itertools
import math
import random

gat_result_dir = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results', 'GAT_results')
if not os.path.exists(gat_result_dir):
    os.makedirs(gat_result_dir)


# class embodying the hyperparameter choice of a GAT model
class HyperparametersGAT(object):

    def __init__(self, updated_params=None):
        self.params = {
            'name': 'GAT',
            'hidden_units': [20, 40, 20],
            'attention_heads': [5, 5, 4],
            'include_ew': True,
            'readout_aggregator': TensorflowGraphGAT.master_node_aggregator,
            'load_specific_data': load_struct_data,
            'batch_size': 32,
            'learning_rate': 0.0005,
            'decay_rate': 0.0005,
            'attn_drop': 0.6,
            'functional_dim': 50,
            'scan_session': 1,
            'nested_CV_level': 'outer',
            'eval_fold_in': 1,
            'eval_fold_out': 4,
            # fixed hyperparameters
            'pers_traits_selection': ['NEO.NEOFAC_A', 'NEO.NEOFAC_C', 'NEO.NEOFAC_E', 'NEO.NEOFAC_N', 'NEO.NEOFAC_O'],
            'use_batch_norm': True,
            'non_linearity': relu,
            'k_outer': 5,
            'k_inner': 5,
            'edgeWeights_filter': lower_bound_filter,
            'pq_threshold': np.inf,
            'train_prog_threshold': 0.1,
            'k_strip_epochs': 5,
            'low_ew_limit': 2.4148,
            'num_epochs': 150}

        # update the default hyper-parameters
        self.update(update_hyper=updated_params)
        if self.params['nested_CV_level'] not in {'inner', 'outer'}:
            raise ValueError('Possbile CV levels: inner, outer')
        # values for the PQ threshold:
        pq_alpha = {
            GATModel.master_node_aggregator: {True: {load_funct_data: 0.01, load_struct_data: 0.01},
                                              False: {load_funct_data: 1.0, load_struct_data: 0.4}},
            GATModel.concat_feature_aggregator: {True: {load_funct_data: 0.4, load_struct_data: 0.4},
                                                 False: {load_funct_data: 1.0, load_struct_data: 0.75}},
            GATModel.average_feature_aggregator: {True: {load_funct_data: 0.5, load_struct_data: 0.5},
                                                  False: {load_funct_data: 1.0, load_struct_data: 1.0}}}
        self.params['pq_threshold'] = pq_alpha[self.params['readout_aggregator']][self.params['include_ew']][
            self.params['load_specific_data']]

        # keep an order on the personality traits predicted at once (so we can decode them from the vectors inferred)
        self.params['pers_traits_selection'] = sorted(self.params['pers_traits_selection'])
        self.params['target_score_type'] = len(self.params['pers_traits_selection'])

    def __str__(self):
        str_dataset = 'GAT_' + self.params['load_specific_data'].__name__.split('_')[1]
        str_dim_sess = 'DIM_' + str(self.params['functional_dim']) + '_' + 'SESS_' + str(self.params['scan_session'])
        str_attn_heads = 'AH_' + ",".join(map(str, self.params['attention_heads']))
        str_hid_units = 'HU_' + ",".join(map(str, self.params['hidden_units']))
        str_traits = 'PT_' + "".join([pers.split('NEO.NEOFAC_')[1] for pers in self.params['pers_traits_selection']])
        str_aggregator = 'AGR_' + self.params['readout_aggregator'].__name__.split('_')[0]
        str_include_ew = 'IW_' + str(self.params['include_ew'])
        str_batch_sz = 'BS_' + str(self.params['batch_size'])
        str_cross_val = 'CV_' + str(self.params['eval_fold_in']) + str(self.params['eval_fold_out']) + self.params[
            'nested_CV_level']
        str_dropout = 'DROP_' + str(self.params['attn_drop'])
        str_learn_rate = 'LR_' + str(self.params['learning_rate'])
        str_decay_rate = 'DR_' + str(self.params['decay_rate'])

        str_params = [str_dataset, str_dim_sess, str_attn_heads, str_hid_units, str_traits, str_aggregator,
                      str_include_ew, str_batch_sz, str_dropout, str_learn_rate, str_decay_rate, str_cross_val]
        if self.params['load_specific_data'].__name__.split('_')[1] == 'struct':
            str_params.remove(str_dim_sess)

        return '_'.join(str_params)

    def get_name(self):
        import re
        return re.compile(re.escape('_CV') + '.*').sub('', re.sub(r"PT_[A-Z]{1,5}_", "", str(self)))

    def update(self, update_hyper):
        if update_hyper is not None:
            self.params.update(update_hyper)

    def checkpoint_file(self):
        return os.path.join(gat_result_dir, 'checkpoint_' + str(self) + '.h5')

    def logs_file(self):
        return os.path.join(gat_result_dir, 'logs_' + str(self) + '.pck')

    def results_file(self):
        return os.path.join(gat_result_dir, 'predictions_' + str(self))

    def proc_data_dir(self):
        if self.params['load_specific_data'] is load_struct_data:
            return dir_structural_data
        else:
            return dir_functional_data

    @staticmethod
    def get_samples_file():
        return os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                            'gat_sampled_models.pck')

    @staticmethod
    def get_sampled_models(max_samples=18000, **kwargs):
        if os.path.exists(HyperparametersGAT.get_samples_file()):
            with open(HyperparametersGAT.get_samples_file(), 'rb') as handle:
                hyparam_choices = pkl.load(handle)
                return hyparam_choices
        choices = {
            'learning_rate': [0.005, 0.001, 0.0005, 0.0001],
            'decay_rate': [0.0005],
            'attn_drop': [0.0, 0.2, 0.4, 0.6, 0.8],
            'readout_aggregator': [GATModel.average_feature_aggregator, GATModel.master_node_aggregator,
                                   GATModel.concat_feature_aggregator],
            'load_specific_data': [load_struct_data, load_funct_data],
            'include_ew': [True, False],
            'batch_size': [32]}
        models_so_far = np.prod(np.array([len(choices[x]) for x in choices.keys()])) * 25
        sampling_left = math.floor(max_samples / models_so_far)
        NO_LAYERS = 3
        sample_ah = list(itertools.product(range(3, 7), repeat=NO_LAYERS))
        sample_hu = list(itertools.product(range(12, 48), repeat=NO_LAYERS))

        def check_feat_expansion(ah_hu_choice):
            for i in range(1, NO_LAYERS - 1):
                if ah_hu_choice[0][i] * ah_hu_choice[1][i] > ah_hu_choice[0][i - 1] * ah_hu_choice[1][i - 1]:
                    return False
            # the last GAT layer averages node features (no multiplication with no of attention heads)
            if ah_hu_choice[1][-1] > ah_hu_choice[0][-2] * ah_hu_choice[1][-2]:
                return False
            return True

        valid_ah_hu = set(filter(lambda ah_hu_choice: check_feat_expansion(ah_hu_choice),
                                 list(itertools.product(sample_ah, sample_hu))))
        choices['arch_width'] = list(map(lambda x: [list(x[0]), list(x[1])], random.sample(valid_ah_hu, sampling_left)))
        with open(HyperparametersGAT.get_samples_file(), 'wb') as handle:
            pickle.dump(choices, handle)

        return choices

    def print_model_details(self):
        params = self.params
        # GAT hyper-parameters
        print('Name of the current GAT model is %s' % self)
        if params['load_specific_data'] == load_struct_data:
            print('Dataset: structural HCP graphs')
        else:
            print('Dataset: functional HCP graphs')
            print('Dimension of graphs: %d and session: %d' % (params['functional_dim'], params['scan_session']))
        print('----- Opt. hyperparams -----')
        print('batch size: ' + str(params['batch_size']))
        print('number of training epochs: ' + str(params['num_epochs']))
        print('lr: ' + str(params['learning_rate']))
        print('l2_coef: ' + str(params['decay_rate']))
        print('droput rate ' + str(params['attn_drop']))
        print('using batch normalization ' + str(params['use_batch_norm']))
        print('----- Archi. hyperparams -----')
        print('nb. layers: ' + str(len(params['hidden_units'])))
        print('nb. units per layer: ' + str(params['hidden_units']))
        print('nb. attention heads: ' + str(params['attention_heads']))
        print('aggregation strategy: ' + str(params['readout_aggregator']))
        print('including edge weights: ' + str(params['include_ew']))
        print('nonlinearity: ' + str(params['non_linearity']))
        print('----- Cross-Validation params. -----')
        print('Nested-CV level: ' + self.params['nested_CV_level'])
        print('Inner split: ' + str(self.params['k_inner']))
        print('Outer split: ' + str(self.params['k_outer']))
        print('Outer evaluation fold id: ' + str(self.params['eval_fold_out']))
        print('Inner evaluation fold id: ' + str(self.params['eval_fold_in']))

    @staticmethod
    def inner_losses(filter_by_params: dict):
        lookup_table = {}
        inner_results = {}
        inner_losses_file = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                                         'gat_inner_eval_losses.pck')
        if os.path.exists(inner_losses_file):
            with open(inner_losses_file, 'rb') as handle:
                inner_results, lookup_table = pkl.load(handle)
        else:
            for result_file in list(filter(lambda x: x.startswith('predictions_'), os.listdir(gat_result_dir))):
                with open(os.path.join(gat_result_dir, result_file), 'rb') as result_fp:
                    results = pkl.load(result_fp)
                if 'params' not in results.keys():
                    log_file = 'logs_' + result_file.split('predictions_')[0]
                    with open(os.path.join(gat_result_dir, log_file), 'rb') as logs_file:
                        params = pkl.load(logs_file)['params']
                    results['params'] = params
                    with open(os.path.join(gat_result_dir, result_file), 'wb') as result_fp:
                        pkl.dump(results, result_fp)
                else:
                    params = results['params']
                config = HyperparametersGAT(params)
                model_name = config.get_name()
                # load the results for a suitable model into main memory
                lookup_table[model_name] = config
                outer_split = config.params['eval_fold_out']
                inner_split = config.params['eval_fold_in']
                if outer_split not in inner_results.keys():
                    inner_results[outer_split] = {}
                if model_name not in inner_results[outer_split].keys():
                    inner_results[outer_split][model_name] = {}
                # save the test losses for the particular inner split (already computed for all traits)
                inner_results[outer_split][model_name][inner_split] = results['test_loss']

            with open(inner_losses_file, 'wb') as handle:
                pkl.dump((inner_results, lookup_table), handle)
        # extract only the evaluation results of the models with specific hyper-parameters
        for out_split in inner_results.keys():
            model_names = list(inner_results[out_split].keys())
            for model in model_names:
                if not filter_by_params.items() <= lookup_table[model].params.items():
                    inner_results[out_split].pop(model)

        return inner_results, lookup_table
