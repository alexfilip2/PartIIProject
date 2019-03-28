from utils.LoadStructuralData import load_baseline_struct_data
import os
from skrvm import RVR
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

baseline_result_dir = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'Results', 'Baselines_results')
if not os.path.exists(baseline_result_dir):
    os.makedirs(baseline_result_dir)


def nested_cross_validation(pers_trait, base_model, hyper_search_space):
    if base_model is LinearRegression:
        model_name = 'LR'
    elif base_model is svm.SVR:
        model_name = 'SVR'
    else:
        model_name = 'RVM'
    result_file = open(os.path.join(baseline_result_dir, 'results_%s' % model_name), 'w')
    # load the dataset
    adjs, scores = load_baseline_struct_data([pers_trait])
    print("The lenght of the dataset is %d" % adjs.shape[0])
    print("The Cross Validation results for personality trait %s are: " % pers_trait)
    # the baseline model to be evaluated
    estimator = base_model()

    inner_cv = KFold(n_splits=5, shuffle=False)
    outer_cv = KFold(n_splits=5, shuffle=False)
    split_id = 1
    for tr_indices, ts_indices in outer_cv.split(adjs):
        inner_X = adjs[tr_indices]
        inner_Y = scores[tr_indices]
        clf = GridSearchCV(estimator=estimator, param_grid=hyper_search_space, cv=inner_cv, verbose=False)
        print(inner_Y.shape, inner_X.shape)
        clf.fit(X=inner_X, y=inner_Y)

        print('Best params for the model %s on split %d are' % (model_name, split_id), file=result_file)
        print(clf.best_params_, file=result_file)
        split_id += 1
        estimator.set_params(**clf.best_params_)
        estimator.fit(X=inner_X, y=inner_Y)
        print('The MSE loss on the outer test set using the best inner model is:', file=result_file)
        print(mean_squared_error(scores[ts_indices], estimator.predict(adjs[ts_indices])), file=result_file)
        print()


if __name__ == "__main__":
    lin_reg = LinearRegression
    svr_reg = svm.SVR
    rvm_reg = RVR

    lin_reg_search = {'fit_intercept': [True, False],
                      'normalize': [True, False]}
    svr_search = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                  'epsilon': [0.1, 0.3, 0.05],
                  'gamma': [0.001, 0.0001],
                  'C': [1.0, 10.0, 100.0]}
    rvm_search = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                  'n_iter': [500, 1000, 1500, 3000],
                  'alpha': [1e-06]}
    nested_cross_validation(pers_trait='NEO.NEOFAC_A', base_model=rvm_reg, hyper_search_space=rvm_search)
