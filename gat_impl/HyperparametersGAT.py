from utils.LoadStructuralData import *
from utils.LoadFunctionalData import *
from gat_impl.TensorflowGraphGAT import *

from tensorflow.nn import relu

checkpts_dir = os.path.join(os.pardir, 'Results', 'GAT_results')
if not os.path.exists(checkpts_dir):
    os.makedirs(checkpts_dir)


# class embodying the hyperparameter choice of a GAT model
class HyperparametersGAT(object):

    def __init__(self, updated_params=None):
        self.params = {
            'hidden_units': [20, 40, 20],
            'attention_heads': [5, 5, 4],
            'include_ew': True,
            'readout_aggregator': TensorflowGraphGAT.master_node_aggregator,
            'load_specific_data': load_struct_data,
            'pers_traits_selection': ['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E'],
            'batch_size': 2,
            'learning_rate': 0.0001,
            'decay_rate': 0.0005,
            'attn_drop': 0.6,
            'functional_dim': 50,
            'scan_session': 1,
            'nested_CV_level': 'outer',
            'eval_fold_in': 1,
            'eval_fold_out': 4,
            # fixed hyperparameters
            'non_linearity': relu,
            'k_outer': 5,
            'k_inner': 5,
            'random_seed': 123,
            'edgeWeights_filter': lower_bound_filter,
            'CHECKPT_PERIOD': 100,
            'gl_tr_prog_threshold': 0.2,
            'no_train_prog':1.0,
            'enough_train_prog': 10.0,
            'k_strip_epochs': 5,
            'low_ew_limit': 2.4148,
            'num_epochs': 200

        }
        self.update(update_hyper=updated_params)
        if self.params['nested_CV_level'] not in {'inner', 'outer'}:
            raise ValueError('Possbile CV levels: inner, outer')
        self.params['target_score_type'] = len(self.params['pers_traits_selection'])

    def __str__(self):
        str_dataset = 'GAT_' + self.params['load_specific_data'].__name__.split('_')[1]
        str_dim_sess = 'DIM_' + str(self.params['functional_dim']) + '_' + 'SESS_' + str(self.params['scan_session'])
        str_attn_heads = 'AH_' + ",".join(map(str, self.params['attention_heads']))
        str_hid_units = 'HU_' + ",".join(map(str, self.params['hidden_units']))
        str_traits = 'PT_' + "".join([pers.split('NEO.NEOFAC_')[1] for pers in self.params['pers_traits_selection']])
        str_aggregator = 'AGR_' + self.params['readout_aggregator'].__name__.split('_')[0]
        str_include_ew = 'IW_' + str(self.params['include_ew'])
        str_limits = 'EL_' + ('No_' if self.params['edgeWeights_filter'] is None else str(self.params['low_ew_limit']))
        str_batch_sz = 'BS_' + str(self.params['batch_size'])
        str_cross_val = 'CV_' + str(self.params['eval_fold_in']) + str(self.params['eval_fold_out']) + self.params[
            'nested_CV_level']
        str_dropout = 'DROP_' + str(self.params['attn_drop'])

        str_params = [str_dataset, str_dim_sess, str_attn_heads, str_hid_units, str_traits, str_aggregator,
                      str_include_ew, str_limits, str_batch_sz, str_cross_val, str_dropout]
        if self.params['load_specific_data'].__name__.split('_')[1] == 'struct':
            str_params.remove(str_dim_sess)
        else:
            str_params.remove(str_limits)

        return '_'.join(str_params)

    def update(self, update_hyper):
        if update_hyper is not None:
            self.params.update(update_hyper)

    def checkpt_file(self):
        return os.path.join(checkpts_dir, 'checkpoint_' + str(self))

    def logs_file(self):
        return os.path.join(checkpts_dir, 'logs_' + str(self))

    def results_file(self):
        return os.path.join(checkpts_dir, 'predictions_' + str(self))

    def proc_data_dir(self):
        if self.params['load_specific_data'] is load_struct_data:
            return dir_proc_struct_data
        else:
            return dir_proc_funct_data

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
        print('l2_coef: ' + str(params['l2_coefficient']))
        print('droput rate ' + str(params['attn_drop']))
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
