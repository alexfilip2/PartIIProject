from utils.LoadStructuralData import load_struct_data, dir_structural_data
from utils.LoadFunctionalData import load_funct_data, dir_functional_data
from sklearn.model_selection import ParameterGrid
from gat_impl.ExecuteGAT import *
from keras.activations import relu
import pickle as pkl
import itertools
import math
import random

gat_result_dir = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results', 'GAT_results')
if not os.path.exists(gat_result_dir):
    os.makedirs(gat_result_dir)


# utility class for storing together the hyper-parameters of a GAT model into an object
class HyperparametersGAT(object):

    def __init__(self, updated_params=None):
        '''
         Initialize the object embodying the configuration of a GAT model.
        :param updated_params: specific hyper-parameters used by current GAT configuration
        '''
        self.params = {
            # architecture hyper-parameters
            'name': 'GAT',
            'hidden_units': [20, 40, 20],
            'attention_heads': [5, 5, 4],
            'include_ew': False,
            'readout_aggregator': TensorflowGraphGAT.master_node_aggregator,
            'use_batch_norm': True,
            'non_linearity': relu,
            # training hyper.
            'load_specific_data': load_struct_data,
            'pers_traits_selection': ['NEO.NEOFAC_A', 'NEO.NEOFAC_C', 'NEO.NEOFAC_E', 'NEO.NEOFAC_N', 'NEO.NEOFAC_O'],
            'learning_rate': 0.0005,
            'decay_rate': 0.0005,
            'attn_drop': 0.6,
            'batch_size': 32,
            'functional_dim': 50,
            'scan_session': 1,
            'num_epochs': 150,
            'pq_threshold': np.inf,
            'train_prog_threshold': 0.1,
            'k_strip_epochs': 5,
            # nested CV hyper.
            'nested_CV_level': 'outer',
            'k_outer': 5,
            'k_inner': 5,
            'eval_fold_out': 4,
            'eval_fold_in': 1}

        # update the default hyper-parameters
        self.update(update_hyper=updated_params)
        if type(self.params['pers_traits_selection']) is not list:
            raise ValueError('The trait selection should be list, not %s' % type(self.params['pers_traits_selection']))
        if self.params['nested_CV_level'] not in {'inner', 'outer'}:
            raise ValueError('Possible CV levels: inner, outer')
        if self.params['name'] != 'GAT':
            raise ValueError('Name of the Graph Attention Network is GAT, not %s' % self.params['name'])
        if len(self.params['attention_heads']) != len(self.params['hidden_units']):
            raise ValueError('Attention heads and hidden units are not specified for the same nr. of layers')
        # values for the PQ threshold:
        pq_thresholds = {GATModel.master_node_aggregator: {True: 0.01, False: 0.5},
                         GATModel.concat_feature_aggregator: {True: 0.5, False: 1.0},
                         GATModel.average_feature_aggregator: {True: 0.5, False: 1.0}}
        self.params['pq_threshold'] = pq_thresholds[self.params['readout_aggregator']][self.params['include_ew']]
        # keep a fixed order on the personality traits so we can decode when predicting them all at once
        self.params['pers_traits_selection'] = sorted(self.params['pers_traits_selection'])
        self.params['target_score_type'] = len(self.params['pers_traits_selection'])

    def __str__(self):
        '''
         Produces a unique string identifier of the current GAT model.
        :return: str of the name of the model, including the nested CV parameters
        '''
        str_dataset = 'GAT_%s' % self.params['load_specific_data'].__name__.split('_')[1]
        str_dim_sess = 'DIM_%dSESS_%d' % (self.params['functional_dim'], self.params['scan_session'])
        str_attn_heads = 'AH_%s' % ",".join(map(str, self.params['attention_heads']))
        str_hid_units = 'HU_%s' % ",".join(map(str, self.params['hidden_units']))
        str_traits = 'PT_%s' % self.get_summarized_traits()
        str_aggregator = 'AGR_%s' % self.params['readout_aggregator'].__name__.split('_')[0]
        str_include_ew = 'IW_%r' % self.params['include_ew']
        str_batch_sz = 'BS_%d' % self.params['batch_size']
        str_dropout = 'DROP_%s' % str(self.params['attn_drop'])
        str_learn_rate = 'LR_%s' % str(self.params['learning_rate'])
        str_decay_rate = 'DR_%s' % str(self.params['decay_rate'])
        str_cross_val = 'CV_%d%d%s' % (self.params['eval_fold_in'], self.params['eval_fold_out'], self.params[
            'nested_CV_level'])

        str_params = [str_dataset, str_dim_sess, str_attn_heads, str_hid_units, str_traits, str_aggregator,
                      str_include_ew, str_batch_sz, str_dropout, str_learn_rate, str_decay_rate, str_cross_val]
        if self.params['load_specific_data'] is load_struct_data:
            str_params.remove(str_dim_sess)
        return '_'.join(str_params)

    def print_model_details(self):
        '''
         Prints the details of the current GAT model as hyper-parameters of architecture, training process and nested CV
        :return: void
        '''
        params = self.params
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

    def get_name(self):
        '''
         Get the name of the GAT model discarding the hyper-parameters of the Nested Cross Validation
        :return: str of the base name of the model
        '''
        import re
        return re.compile(re.escape('_CV') + '.*').sub('', re.sub(r"PT_[A-Z]{1,5}_", "", str(self)))

    def get_summarized_traits(self):
        '''
         Summarize the names of the traits targeted at once.
        :return: str of the concatenation of the trait names without the common prefixes
        '''
        return ''.join(self.params['pers_traits_selection']).replace('NEO.NEOFAC_', '')

    def update(self, update_hyper):
        '''
         Updates the default hyper-parameters of the GAT configuration object
        :param update_hyper: dict of new hyper-parameters
        :return: void, it's changing the internal state of the object
        '''
        if update_hyper is not None:
            self.params.update(update_hyper)

    def checkpoint_file(self):
        '''
        Retrieves the path to the checkpoint file where the model (its parameters) is/should be saved
        :return: str path
        '''
        return os.path.join(gat_result_dir, 'checkpoint_' + str(self) + '.h5')

    def logs_file(self):
        '''
        Retrieves the path to the logs file where the training history of the model is/should be saved
        :return: str path
        '''
        return os.path.join(gat_result_dir, 'logs_' + str(self) + '.pck')

    def results_file(self):
        '''
         Retrieves the path to the results file where the evaluation data: test loss, predictions is/should be saved
        :return: str path
        '''
        return os.path.join(gat_result_dir, 'predictions_' + str(self))

    def get_results(self):
        '''
        Retrieve the results of the model.
        :return: dict with evaluation results: losses, metrics, predictions
        '''
        results = None
        if os.path.exists(self.results_file()):
            with open(self.results_file(), 'rb') as result_fp:
                results = pkl.load(result_fp)
        return results

    def processed_data_dir(self):
        '''
         Retrieves the path to the directory where the dataset used by this GAT model is stored
        :return: str path
        '''
        if self.params['load_specific_data'] is load_struct_data:
            return dir_structural_data
        else:
            return dir_functional_data

    @staticmethod
    def get_sampled_models(max_samples=18000, no_layers=3, **kwargs):
        '''
         Samples a pre-defined number of GAT configurations for the inner CV of the nested CV phase
        :param max_samples: maximum number of sampled models
        :param kwargs: compatibility with the sampling function of baseline models
        :param no_layers: number of layers of the sampled configurations
        :return: dict of hyper-parameters choices to be converted to a Grid Search
        '''
        samples_file = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                                    'gat_sampled_models.pck')
        if os.path.exists(samples_file):
            with open(samples_file, 'rb') as handle:
                return pkl.load(handle)

        choices = {
            'learning_rate': [0.005, 0.001, 0.0005, 0.0001],
            'decay_rate': [0.0005],
            'attn_drop': [0.0, 0.2, 0.4, 0.6, 0.8],
            'readout_aggregator': [GATModel.average_feature_aggregator, GATModel.master_node_aggregator,
                                   GATModel.concat_feature_aggregator],
            'load_specific_data': [load_struct_data, load_funct_data],
            'include_ew': [True, False],
            'batch_size': [32]}
        models_so_far = np.prod(np.array([len(choices[x]) for x in choices.keys()])) * 5 * 5
        sampling_left = math.floor(max_samples / models_so_far)
        sample_ah = list(itertools.product(range(3, 7), repeat=no_layers))
        sample_hu = list(itertools.product(range(12, 48), repeat=no_layers))

        def check_feat_expansion(ah_hu_choice):
            '''
             Checks if the particular choice of attention heads and units per GAT layer follows an expansion approach
             of the node features' dimensionality
            :param ah_hu_choice: tuple of two lists of the choices of attention heads and hidden units
            :return: bool, the validity of the choice
            '''
            for i in range(1, no_layers - 1):
                if ah_hu_choice[0][i] * ah_hu_choice[1][i] > ah_hu_choice[0][i - 1] * ah_hu_choice[1][i - 1]:
                    return False
            # the last GAT layer averages node features (no multiplication with no of attention heads)
            if ah_hu_choice[1][-1] > ah_hu_choice[0][-2] * ah_hu_choice[1][-2]:
                return False
            return True

        valid_ah_hu = set(filter(lambda ah_hu_choice: check_feat_expansion(ah_hu_choice),
                                 list(itertools.product(sample_ah, sample_hu))))
        choices['arch_width'] = list(map(lambda x: [list(x[0]), list(x[1])], random.sample(valid_ah_hu, sampling_left)))
        with open(samples_file, 'wb') as handle:
            pickle.dump(choices, handle)

        return choices

    @staticmethod
    def inner_losses(filter_by_params: dict):
        '''
         Extract the inner CV test losses obtained on the inner evaluation folds for all the models sampled using
         the values of the hyper-parameters as specified by the filter dict.
        :param filter_by_params: dict of hyper-parameter name and filtering value
        :return: dict of individual model test losses
        '''
        lookup_table = {}
        inner_results = {}
        inner_losses_file = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                                         'gat_inner_eval_losses.pck')
        if os.path.exists(inner_losses_file):
            with open(inner_losses_file, 'rb') as fp:
                inner_results, lookup_table = pkl.load(fp)
        else:
            # retrieve only the results of the sampled models
            grid = ParameterGrid(HyperparametersGAT.get_sampled_models())
            ncv_params = {'k_outer': HyperparametersGAT().params['k_outer'],
                          'k_inner': HyperparametersGAT().params['k_inner'],
                          'nested_CV_level': 'inner'}
            for params in grid:
                params['attention_heads'] = params['arch_width'][0]
                params['hidden_units'] = params['arch_width'][1]
                gat_model_config = HyperparametersGAT(params)
                gat_model_config.update(ncv_params)
                for eval_out in range(ncv_params['k_outer']):
                    gat_model_config.params['eval_fold_out'] = eval_out
                    inner_results[eval_out] = {}
                    for eval_in in range(ncv_params['k_inner']):
                        gat_model_config.params['eval_fold_in'] = eval_in
                        results = gat_model_config.get_results()
                        if not results:
                            print('Incomplete nested CV results, missing model %s' % gat_model_config)
                            return
                        model_name = gat_model_config.get_name()
                        lookup_table[model_name] = gat_model_config
                        if model_name not in inner_results[eval_out].keys():
                            inner_results[eval_out][model_name] = {}
                        inner_results[eval_out][model_name][eval_in] = results['test_loss']

        # save the unfiltered inner losses
        with open(inner_losses_file, 'wb') as handle:
            pkl.dump((inner_results, lookup_table), handle)

        # extract only the evaluation results of the models with specific hyper-parameter values
        for out_split in inner_results.keys():
            model_names = list(inner_results[out_split].keys())
            for model in model_names:
                if not filter_by_params.items() <= lookup_table[model].params.items():
                    inner_results[out_split].pop(model)
        return inner_results, lookup_table
