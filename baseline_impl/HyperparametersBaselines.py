from utils.LoadStructuralData import *
from utils.LoadFunctionalData import *
from skrvm import RVR
from sklearn import svm
from sklearn.linear_model import LinearRegression
import inspect


baseline_result_dir = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                                   'Baselines_results')
if not os.path.exists(baseline_result_dir):
    os.makedirs(baseline_result_dir)


# class embodying the hyperparameter choice of a GAT model
class HyperparametersBaselines(object):

    def __init__(self, updated_params=None):
        self.params = {
            'name': 'RVM',
            'load_specific_data': load_struct_data,
            'pers_traits_selection': ['NEO.NEOFAC_A'],
            'functional_dim': 50,
            'scan_session': 1,
            'kernel': 'rbf',
            'epsilon': 0.1,
            'gamma': 0.001,
            'C': 1.0,
            'n_iter': 500,
            'alpha': 1e-06,
            'fit_intercept': True,
            'normalize': True,
            # nested cross validation parameters
            'nested_CV_level': 'outer',
            'eval_fold_in': 1,
            'eval_fold_out': 4,
            # fixed hyperparameters
            'k_outer': 5,
            'k_inner': 5,
            'edgeWeights_filter': lower_bound_filter,
            'low_ew_limit': 2.4148}

        # update the default hyper-parameters
        self.update(update_hyper=updated_params)
        if self.params['name'] == 'LR':
            self.params['model'] = LinearRegression
        elif self.params['name'] == 'SVR':
            self.params['model'] = svm.SVR
        elif self.params['name'] == 'RVM':
            self.params['model'] = RVR
        else:
            raise ValueError('Possbile Baselines Models: LR, SVR or RVM ')

        if self.params['nested_CV_level'] not in {'inner', 'outer'}:
            raise ValueError('Possbile CV levels: inner, outer')
        self.params['dataset'] = 'struct' if self.params['load_specific_data'] == load_struct_data else 'funct'

    def __str__(self):
        str_dataset = self.params['name'] + '_' + self.params['dataset']
        str_dim_sess = 'DIM_' + str(self.params['functional_dim']) + '_' + 'SESS_' + str(self.params['scan_session'])
        str_traits = 'PT_' + self.params['pers_traits_selection'][0].split('_')[1]
        str_cross_val = 'CV_' + str(self.params['eval_fold_in']) + str(self.params['eval_fold_out']) + self.params[
            'nested_CV_level']
        str_params = [str_dataset, str_dim_sess, str_traits]
        for arg in sorted(list(self.params.keys())):
            if arg in inspect.getfullargspec(self.params['model']).args:
                str_params.append(arg + '_' + str(self.params[arg]))
        str_params.append(str_cross_val)
        if self.params['dataset'] == 'struct':
            str_params.remove(str_dim_sess)

        return '_'.join(str_params)

    def get_suitable_args(self):
        args = {}
        for arg in sorted(list(self.params.keys())):
            if arg in inspect.getfullargspec(self.params['model']).args:
                args[arg] = self.params[arg]
        return args

    def get_name(self):
        import re
        return re.compile(re.escape('_CV') + '.*').sub('', re.sub(r"PT_[A-Z]_", "", str(self)))

    def update(self, update_hyper):
        if update_hyper is not None:
            self.params.update(update_hyper)

    def results_file(self):
        return os.path.join(baseline_result_dir, 'predictions_' + str(self))

    @staticmethod
    def get_sampled_models(baseline_name, **kwargs):
        if baseline_name == 'LR':
            search_space = {'name': [baseline_name],
                            'fit_intercept': [True, False],
                            'normalize': [True, False]}
        elif baseline_name == 'SVR':
            search_space = {'name': [baseline_name],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                            'epsilon': [0.1, 0.3, 0.5],
                            'gamma': [1.0, 0.1, 0.001, 0.0001],
                            'C': [1, 10, 100]}
        elif baseline_name == 'RVM':
            search_space = {'name': [baseline_name],
                            'kernel': ['rbf', 'linear'],
                            'n_iter': [100, 500, 750, 1000],
                            'alpha': [1e-06, 1e-05]}
        else:
            print('Wrong choice of model name')
            return
        data_type = {'load_specific_data': [load_funct_data, load_struct_data],
                     'pers_traits_selection': [['NEO.NEOFAC_A'], ['NEO.NEOFAC_C'], ['NEO.NEOFAC_E'], ['NEO.NEOFAC_N'],
                                               ['NEO.NEOFAC_O']]}
        return {**search_space, **data_type}

    @staticmethod
    def inner_losses(filter_by_params: dict):
        lookup_table = {}
        inner_results = {}
        inner_losses_file = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Results',
                                         'baseline_inner_eval_losses.pck')
        if os.path.exists(inner_losses_file):
            with open(inner_losses_file, 'rb') as handle:
                inner_results, lookup_table = pkl.load(handle)
        else:
            for result_file in os.listdir(baseline_result_dir):
                if result_file.startswith('predictions_'):
                    # load the results for a suitable model into main memory
                    with open(os.path.join(baseline_result_dir, result_file), 'rb') as out_result_file:
                        results = pkl.load(out_result_file)
                        config_obj = results['config']
                        model_name = config_obj.get_name()
                        # fill in the lookup table
                        lookup_table[model_name] = config_obj
                        outer_split = config_obj.params['eval_fold_out']
                        inner_split = config_obj.params['eval_fold_in']
                        trait, = config_obj.params['pers_traits_selection']
                        if outer_split not in inner_results.keys():
                            inner_results[outer_split] = {}
                        if model_name not in inner_results[outer_split].keys():
                            inner_results[outer_split][model_name] = {}
                        if inner_split not in inner_results[outer_split][model_name].keys():
                            inner_results[outer_split][model_name][inner_split] = {}
                        # save the test losses for the particular inner split
                        inner_results[outer_split][model_name][inner_split][trait] = results['test_loss'][trait]
            with open(inner_losses_file, 'wb') as handle:
                pkl.dump((inner_results, lookup_table), handle)
        # extract only the evaluation results of the models with specific hyper-parameters
        for out_split in inner_results.keys():
            model_names = list(inner_results[out_split].keys())
            for model in model_names:
                if not filter_by_params.items() <= lookup_table[model].params.items():
                    inner_results[out_split].pop(model)
        return inner_results, lookup_table
