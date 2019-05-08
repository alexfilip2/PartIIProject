from utils.LoadStructuralData import load_struct_data
from utils.LoadFunctionalData import load_funct_data
from skrvm import RVR
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import inspect
import os
import pickle as pkl

cached_data = {}
baseline_result_dir = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                                   'Baselines_results')
if not os.path.exists(baseline_result_dir):
    os.makedirs(baseline_result_dir)


# utility class for storing together the hyper-parameters of a baseline model into an object
class HyperparametersBaselines(object):
    def __init__(self, updated_params=None):
        '''
        Initialize the object embodying the configuration of a baseline model.
        :param updated_params: specific hyper-parameters used by current baseline configuration
        '''
        self.params = {
            # model hyper-parameters
            'name': 'RVM',
            'kernel': 'rbf',
            'epsilon': 0.1,
            'gamma': 0.001,
            'C': 1.0,
            'alpha': 1.0,
            'n_iter': 500,
            'fit_intercept': True,
            'normalize': True,
            # training hyper.
            'load_specific_data': load_struct_data,
            'pers_traits_selection': ['NEO.NEOFAC_A'],
            'functional_dim': 50,
            'scan_session': 1,
            # nested CV hyper.
            'nested_CV_level': 'outer',
            'eval_fold_in': 1,
            'eval_fold_out': 4,
            'k_outer': 5,
            'k_inner': 5}

        # update the default hyper-parameters
        self.update(update_hyper=updated_params)
        if self.params['name'] == 'LR':
            self.params['model'] = Ridge
        elif self.params['name'] == 'SVR':
            self.params['model'] = SVR
        elif self.params['name'] == 'RVM':
            self.params['model'] = RVR
        else:
            raise ValueError('Possible Baselines Models: LR, SVR or RVM ')
        if self.params['nested_CV_level'] not in {'inner', 'outer'}:
            raise ValueError('Possible CV levels: inner, outer')

    def __str__(self):
        '''
         Produces a unique string identifier of the current baseline model.
        :return: str of the name of the model, including the nested CV parameters
        '''
        str_dataset = self.params['name'] + '_' + self.params['load_specific_data'].__name__.split('_')[1]
        str_dim_sess = 'DIM_%d_SESS_%d' % (self.params['functional_dim'], self.params['scan_session'])
        str_traits = 'PT_%s' % ''.join(self.params['pers_traits_selection']).replace('NEO.NEOFAC_', '')
        str_cross_val = 'CV_%d%d%s' % (self.params['eval_fold_in'], self.params['eval_fold_out'], self.params[
            'nested_CV_level'])
        str_params = [str_dataset, str_dim_sess, str_traits]
        # filter just the hyperparameters used by the current baseline when printing its name
        for arg in sorted(list(self.params.keys())):
            if arg in inspect.getfullargspec(self.params['model']).args:
                str_params.append(arg + '_' + str(self.params[arg]))
        str_params.append(str_cross_val)
        if self.params['load_specific_data'] is load_struct_data:
            str_params.remove(str_dim_sess)

        return '_'.join(str_params)

    def get_suitable_args(self):
        '''
         Inspect the signature of the constructor of the baseline class used and extract its formal parameter names.
        :return: dict of hyperparameters and their values specific to the current baseline
        '''
        args = {}
        for arg in sorted(list(self.params.keys())):
            if arg in inspect.getfullargspec(self.params['model']).args:
                args[arg] = self.params[arg]
        return args

    def get_name(self):
        '''
         Get the name of the baseline model discarding the hyper-parameters of the Nested Cross Validation.
        :return: str of the base name of the model
        '''
        import re
        return re.compile(re.escape('_CV') + '.*').sub('', re.sub(r"PT_[A-Z]_", "", str(self)))

    def update(self, update_hyper):
        '''
         Updates the default hyper-parameters of the baseline configuration object
        :param update_hyper: dict of new hyper-parameters
        :return: void, it's changing the internal state of the object
        '''
        if update_hyper is not None:
            self.params.update(update_hyper)

    def load_data(self):
        global cached_data
        loader_data = self.params['load_specific_data']
        trait_choice, = self.params['pers_traits_selection']
        if loader_data in cached_data.keys():
            if trait_choice in cached_data[loader_data].keys():
                return cached_data[loader_data][trait_choice]
            else:
                uncached_data = loader_data(self.params)
                cached_data[loader_data][trait_choice] = uncached_data
                return uncached_data
        else:
            cached_data[loader_data] = {}
            uncached_data = loader_data(self.params)
            cached_data[loader_data][trait_choice] = uncached_data
            return uncached_data

    def results_file(self):
        '''
         Retrieves the path to the results file where the evaluation data: test loss, predictions is/should be saved
        :return: str path
        '''
        return os.path.join(baseline_result_dir, 'predictions_' + str(self))

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

    @staticmethod
    def get_sampled_models(baseline_name, **kwargs):
        '''
          Samples a pre-defined number of baseline configurations for the inner CV of the nested CV phase.
        :param baseline_name: str name of the baseline
        :param kwargs: compatibility the the sampling function og GAT hyperparameters
        :return: dict of hyper-parameters choices to be converted to a Grid Search
        '''
        if baseline_name == 'LR':
            search_space = {'name': [baseline_name],
                            'fit_intercept': [False],
                            'normalize': [False],
                            'solver ': ['auto'],
                            'alpha': [1.0, 0.5, 0.1]}
        elif baseline_name == 'SVR':
            search_space = {'name': [baseline_name],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                            'epsilon': [0.1, 0.3, 0.5],
                            'gamma': [1.0, 0.1, 0.001, 0.0001],
                            'C': [1, 10, 100]}
        elif baseline_name == 'RVM':
            search_space = {'name': [baseline_name],
                            'kernel': ['rbf', 'linear'],
                            'n_iter': [10, 50, 100],
                            'alpha': [1e-06, 1e-05]}
        else:
            print('Wrong choice of model name')
            return
        data_type = {'load_specific_data': [load_funct_data, load_struct_data],
                     'pers_traits_selection': [['NEO.NEOFAC_A'], ['NEO.NEOFAC_C'], ['NEO.NEOFAC_E'], ['NEO.NEOFAC_N'],
                                               ['NEO.NEOFAC_O']]}
        return {**search_space, **data_type}
