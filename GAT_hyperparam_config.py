from BaseGAT import *


# class embodying the hyperparameter choice of a GAT model
class GAT_hyperparam_config(object):

    def __init__(self, updated_params=None):
        self.params = {
            'hidden_units': [20, 40, 20],
            'attention_heads': [5, 5, 4],
            'include_ew': True,
            'readout_aggregator': BaseGAT.master_node_aggregator,
            'num_epochs': 10000,
            'load_specific_data': load_struct_data,
            'pers_traits_selection': ['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E'],
            'batch_size': 2,
            'edgeWeights_filter': None,
            'patience': 25,
            'CHECKPT_PERIOD': 25,
            'learning_rate': 0.0001,
            'l2_coefficient': 0.0005,
            'residual': False,
            'attn_drop': 0.6,
            'ffd_drop': 0.6,
            'non_linearity': tf.nn.relu,
            'random_seed': 123,
            'functional_dim': 50,
            'scan_session': 1,
            'eval_fold_in': 1,
            'eval_fold_out': 4,
            'k_outer': 5,
            'k_inner': 5,
            'nested_CV_level': 'outer'

        }
        self.update(update_hyper=updated_params)
        if self.params['nested_CV_level'] not in {'inner', 'outer'}:
            raise ValueError('Possbile CV levels: inner, outer')
        self.checkpts_dir = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'GAT_checkpoints')
        if not os.path.exists(self.checkpts_dir):
            os.makedirs(self.checkpts_dir)

    def __str__(self):
        str_traits = 'PT_' + "".join([pers.split('NEO.NEOFAC_')[1] for pers in self.params['pers_traits_selection']])
        str_attn_heads = 'AH_' + ",".join(map(str, self.params['attention_heads']))
        str_hid_units = 'HU_' + ",".join(map(str, self.params['hidden_units']))
        str_aggregator = 'AGR_' + self.params['readout_aggregator'].__name__.split('_')[0]
        str_limits = 'EL_' + ('None' if self.params['edgeWeights_filter'] is None else str(self.params['ew_limits']))
        str_batch_sz = '_BS_' + str(self.params['batch_size'])
        str_dataset = 'GAT_' + self.params['load_specific_data'].__name__.split('_')[1]
        str_include_ew = 'IW_' + str(self.params['include_ew'])
        str_cross_val = 'CV_' + str(self.params['eval_fold_in']) + str(self.params['eval_fold_out']) + self.params[
            'nested_CV_level']

        return '_'.join([str_dataset, str_attn_heads, str_hid_units, str_traits, str_aggregator, str_include_ew,
                         str_limits, str_batch_sz, str_cross_val])

    def update(self, update_hyper):
        if update_hyper is not None:
            self.params.update(update_hyper)

    def checkpt_file(self):
        return os.path.join(self.checkpts_dir, 'checkpoint_' + str(self))

    def proc_data_dir(self):
        if self.params['load_specific_data'] is load_struct_data:
            return dir_proc_struct_data
        else:
            return dir_proc_funct_data

    def logs_file(self):
        return os.path.join(self.checkpts_dir, 'logs_' + str(self))

    def print_model_details(self):
        params = self.params
        # GAT hyper-parameters
        batch_sz = params['batch_size']  # batch training size
        nb_epochs = params['num_epochs']  # number of learning iterations over the trainign dataset
        lr = params['learning_rate']  # learning rate
        l2_coef = params['l2_coefficient']  # weight decay
        hid_units = params['hidden_units']  # numbers of features produced by each attention head per network layer
        n_heads = params['attention_heads']  # number of attention heads on each layer
        residual = params['residual']
        nonlinearity = params['non_linearity']
        aggregator = params['readout_aggregator']
        include_weights = params['include_ew']

        print('Name of the current GAT model is %s' % self)
        print('Dataset:' + ' HCP graphs')
        print('----- Opt. hyperparams -----')
        print('batch size: ' + str(batch_sz))
        print('number of training epochs: ' + str(nb_epochs))
        print('lr: ' + str(lr))
        print('l2_coef: ' + str(l2_coef))
        print('----- Archi. hyperparams -----')
        print('nb. layers: ' + str(len(hid_units)))
        print('nb. units per layer: ' + str(hid_units))
        print('nb. attention heads: ' + str(n_heads))
        print('aggregation strategy: ' + str(aggregator))
        print('including edge weights: ' + str(include_weights))
        print('residual: ' + str(residual))
        print('nonlinearity: ' + str(nonlinearity))
