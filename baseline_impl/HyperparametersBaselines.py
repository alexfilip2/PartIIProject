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
            'model': RVR,
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
        if self.params['model'] is LinearRegression:
            self.params['name'] = 'LR'
        elif self.params['model'] is svm.SVR:
            self.params['name'] = 'SVR'
        else:
            self.params['name'] = 'RVM'
        if self.params['nested_CV_level'] not in {'inner', 'outer'}:
            raise ValueError('Possbile CV levels: inner, outer')

    def __str__(self):
        str_dataset = self.params['name'] + '_' + self.params['load_specific_data'].__name__.split('_')[1]
        str_dim_sess = 'DIM_' + str(self.params['functional_dim']) + '_' + 'SESS_' + str(self.params['scan_session'])
        str_traits = 'PT_' + self.params['pers_traits_selection'][0].split('_')[1]
        str_cross_val = 'CV_' + str(self.params['eval_fold_in']) + str(self.params['eval_fold_out']) + self.params[
            'nested_CV_level']
        str_params = [str_dataset, str_dim_sess, str_traits]
        for arg in sorted(list(self.params.keys())):
            if arg in inspect.getfullargspec(self.params['model']).args:
                str_params.append(arg + '_' + str(self.params[arg]))
        str_params.append(str_cross_val)
        if self.params['load_specific_data'].__name__.split('_')[1] == 'struct':
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
