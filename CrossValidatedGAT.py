import time
import pickle
import multiprocessing

from MainGAT import *
from GAT_hyperparam_config import GAT_hyperparam_config


class CrossValidatedGAT(MainGAT):
    @classmethod
    def default_params(cls):
        return GAT_hyperparam_config()

    def __init__(self, args):
        # Load the GAT_hyperparam_config object of the current model
        if args is None:
            self.config = self.default_params()
        else:
            self.config = args
        # Load the hyperparameter configuration of the current model
        self.params = self.config.params
        # Print the model details
        self.config.print_model_details()
        # Initialize the model skeleton, the computation graph and the parameters
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = multiprocessing.cpu_count()
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph, config=config)

        with self.graph.as_default():
            tf.set_random_seed(self.params['random_seed'])
            self.placeholders = {}
            self.ops = {}
            self.load_CV_data()
            self.split_nested_CV()
            self.make_model()
            # Restore/initialize variables:
            if os.path.exists(self.config.checkpt_file()) and os.path.exists(self.config.logs_file()):
                self.restore_model()
            else:
                self.initialize_model()

    # Load the entire dataset structural or functional that will be used
    def load_CV_data(self):
        # Load data:
        # dictionary of adjancency, node features, bias matrices, personality scores indexed by subject ID's
        self.data, self.subjects = self.params['load_specific_data'](self.params)
        # nr of nodes of each graph
        self.nb_nodes = self.data[self.subjects[0]]['adj'].shape[-1]
        # the initial dimension F of each node's feature vector
        self.ft_size = self.data[self.subjects[0]]['feat'].shape[-1]
        # the number of the big-five personality traits the model is targeting at once
        self.outGAT_sz_target = len(self.params['pers_traits_selection'])

    # Split the entire dataset for a particular training and testing of the nested Cross Validation
    def split_nested_CV(self):
        eval_fold_in = self.params['eval_fold_in']  # the specific inner fold chosen for evaluation
        eval_fold_out = self.params['eval_fold_out']  # the specific outer fold chosen for evaluation
        k_outer = self.params['k_outer']  # the number of outer folds
        k_inner = self.params['k_inner']  # the number of inner folds
        # prepare the outer split
        outer_fold_size = len(self.subjects) // k_outer
        self.outer_train = np.concatenate((self.subjects[:outer_fold_size * eval_fold_out],
                                           self.subjects[outer_fold_size * (eval_fold_out + 1):]))
        self.outer_test = self.subjects[outer_fold_size * eval_fold_out:outer_fold_size * (eval_fold_out + 1)]
        # prepare the inner split
        inner_fold_size = len(self.outer_train) // k_inner
        self.inner_train = np.concatenate((self.outer_train[:inner_fold_size * eval_fold_in],
                                           self.outer_train[inner_fold_size * (eval_fold_in + 1):]))
        self.inner_test = self.outer_train[inner_fold_size * eval_fold_in:inner_fold_size * (eval_fold_in + 1)]

        assert set(self.inner_train).isdisjoint(set(self.inner_test))
        assert set(self.outer_train).isdisjoint(set(self.outer_test))

    def make_model(self):
        with tf.variable_scope('input'):
            self.placeholders['ftr_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.nb_nodes, self.ft_size))
            self.placeholders['bias_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.nb_nodes, self.nb_nodes))
            self.placeholders['score_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.outGAT_sz_target))
            self.placeholders['adj_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.nb_nodes, self.nb_nodes))
            self.placeholders['include_ew'] = tf.placeholder(dtype=tf.bool, shape=())
            prediction, self.ops['unif_loss'], self.ops['excl_loss'] = \
                MainGAT.inference_keras(self, in_feat_vects=self.placeholders['ftr_in'],
                                        adj_mat=self.placeholders['adj_in'],
                                        bias_mat=self.placeholders['bias_in'],
                                        include_weights=self.placeholders['include_ew'],
                                        hid_units=self.params['hidden_units'],
                                        n_heads=self.params['attention_heads'],
                                        target_score_type=self.outGAT_sz_target,
                                        aggregator=self.params['readout_aggregator'],
                                        residual=self.params['residual'],
                                        activation=self.params['non_linearity'],
                                        attn_drop=self.params['attn_drop'],
                                        ffd_drop=self.params['ffd_drop'])

            self.ops['loss'] = tf.losses.mean_squared_error(labels=self.placeholders['score_in'],
                                                            predictions=prediction)

            # minibatch update operations
            self.ops['zero_grads_ops'], self.ops['accum_ops'], self.ops['apply_ops'] = super().batch_training(
                loss=self.ops['loss'], u_loss=self.ops['unif_loss'], e_loss=self.ops['excl_loss'],
                lr=self.params['learning_rate'], l2_coef=self.params['l2_coefficient'])

    # run the list of Operation objects using the data of the subject with different dropout rate for training
    def feed_forward_op(self, ops, subj_key, is_train):
        if is_train:
            self.params['attn_drop'] = self.params['ffd_drop'] = 0.6
        else:
            self.params['attn_drop'] = self.params['ffd_drop'] = 0.0

        subj_data = self.data[subj_key]
        return self.sess.run(ops, feed_dict={self.placeholders['ftr_in']: subj_data['feat'],
                                             self.placeholders['bias_in']: subj_data['bias'],
                                             self.placeholders['score_in']: subj_data['score'],
                                             self.placeholders['adj_in']: subj_data['adj'],
                                             self.placeholders['include_ew']: self.params['include_ew']})

    # run one batch of training: update the weights of GAT and calculate the loss of the batch of examples
    def batch_train_step(self, iteration, train_subjs):
        batch_avg_loss, batch_avg_uloss, batch_avg_eloss = 0.0, 0.0, 0.0
        # reset gradients to 0 before entering minibatch training loop
        self.sess.run(self.ops['zero_grads_ops'])
        # loop over the mini-batch and execute accumulate-gradient operation
        for batch_step in range(self.params['batch_size']):
            index = batch_step + iteration * self.params['batch_size']
            self.feed_forward_op(ops=[self.ops['accum_ops']], subj_key=train_subjs[index], is_train=True)

        # after finished looping over the batch, apply gradients
        self.sess.run(self.ops['apply_ops'])

        # Calculate the validation loss after every single batch training
        for batch_step in range(self.params['batch_size']):
            index = batch_step + iteration * self.params['batch_size']
            (expl_loss, expl_u_loss, expl_e_loss) = self.feed_forward_op(ops=[self.ops['loss'],
                                                                              self.ops['unif_loss'],
                                                                              self.ops['excl_loss']],
                                                                         subj_key=train_subjs[index],
                                                                         is_train=True)
            batch_avg_loss += expl_loss
            batch_avg_uloss += expl_u_loss
            batch_avg_eloss += expl_e_loss

        return map(lambda x: x / self.params['batch_size'], [batch_avg_loss, batch_avg_uloss, batch_avg_eloss])

    # run one batch of training: on training_set which comes from a k_split of the dataset (inner or outer)
    def run_epoch_training(self, training_set, k_split):
        # Train loop
        # number of training examples for the current phase of nested CV (inner or outer)
        tr_size = len(training_set) // k_split * (
                k_split - 1)  # make a training:validation:test ration of k_split-2:1:1
        # number of iterations of the training set on batch-training
        tr_iterations = tr_size // self.params['batch_size']
        # Array for logging the training loss, the uniform loss, the exclusive loss
        tr_loss_log = np.zeros(tr_iterations)
        tr_uloss_log = np.zeros(tr_iterations)
        tr_eloss_log = np.zeros(tr_iterations)
        # shuffle the training dataset
        shuf_subjs = shuffle_tr_data(training_set, tr_size)
        for iteration in range(tr_iterations):
            tr_loss_log[iteration], tr_uloss_log[iteration], tr_eloss_log[iteration] = self.batch_train_step(
                iteration=iteration, train_subjs=shuf_subjs)

        return map(lambda x: np.sum(x) / tr_iterations, [tr_loss_log, tr_uloss_log, tr_eloss_log])

    def run_epoch_validation(self, training_set, k_split):
        # number of training, validation, test graph examples
        vl_size = len(training_set) // k_split
        tr_size = len(training_set) - vl_size
        vl_avg_loss = 0.0
        for vl_step in range(tr_size, tr_size + vl_size):
            (vl_expl_loss,) = self.feed_forward_op([self.ops['loss']], subj_key=training_set[vl_step], is_train=False)
            vl_avg_loss += vl_expl_loss

        vl_avg_loss /= vl_size
        return vl_avg_loss

    def train(self):
        if self.trained_flag:
            return
        # choose the correct training set of subjects
        if self.params['nested_CV_level'] == 'inner':
            training_set = self.inner_train
            k_split = self.params['k_inner']
        else:
            training_set = self.outer_train
            k_split = self.params['k_outer']

        ts_size = len(training_set) // (k_split - 1)
        vl_size = len(training_set) // k_split
        tr_size = len(training_set) - vl_size
        print('The training size is: %d, the validation: %d and the test: %d' % (tr_size, vl_size, ts_size))
        # record the minimum validation loss encountered until current epoch
        vlss_mn = np.inf
        # store the validation loss of previous epoch
        vlss_early_model = np.inf
        # record the number of consecutive epochs when the loss doesn't improve
        curr_step = 0

        for epoch in range(self.last_epoch, self.params['num_epochs']):
            total_time_start = time.time()
            epoch_tr_loss, epoch_uloss, epoch_eloss = self.run_epoch_training(training_set=training_set,
                                                                              k_split=k_split)
            epoch_val_loss = self.run_epoch_validation(training_set=training_set, k_split=k_split)
            self.logs[epoch] = {"tr_loss": epoch_tr_loss,
                                "val_loss": epoch_val_loss,
                                "u_loss": epoch_uloss,
                                "e_loss": epoch_eloss,
                                }
            epoch_time = time.time() - total_time_start
            print('Training: loss = %.5f | Val: loss = %.5f | '
                  'Unifrom loss: %f| Exclusive loss: %f | '
                  'Elapsed epoch time: %.5f' % (epoch_tr_loss, epoch_val_loss, epoch_uloss, epoch_eloss, epoch_time))
            self.test()
            if epoch % self.params['CHECKPT_PERIOD'] == 0:
                self.save_model(last_epoch=epoch + 1, fully_trained=False)

                print("Training progress after %d epochs saved in path: %s" % (epoch, self.config.checkpt_file()))

            # wait for the validation loss to settle before the specified number of training iterations
            if epoch_val_loss <= vlss_mn:
                vlss_early_model = epoch_val_loss
                vlss_mn = np.min((epoch_val_loss, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == self.params['patience']:
                    self.save_model(last_epoch=epoch, fully_trained=True)
                    print("Training progress after %d epochs saved in path: %s" % (epoch, self.config.checkpt_file()))
                    print('Early stop! Min loss: ', vlss_mn)
                    print('Early stop model validation loss: ', vlss_early_model)
                    break

    def test(self):
        # choose the correct training set of subjects
        if self.params['nested_CV_level'] == 'inner':
            test_set = self.inner_test
        else:
            test_set = self.outer_test
        # number of training, validation, test graph examples
        ts_size = len(test_set)
        ts_avg_loss = 0.0
        for ts_step in range(ts_size):
            (ts_example_loss,) = self.feed_forward_op([self.ops['loss']], subj_key=test_set[ts_step], is_train=False)
            ts_avg_loss += ts_example_loss

        ts_avg_loss /= ts_size
        print('Test: loss = %.5f for the model %s' % (ts_avg_loss, self.config))
        # self.sess.close()

    def save_model(self, last_epoch: int, fully_trained: bool = False) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        parameters_to_save = {
            "params": self.params,
            "weights": weights_to_save,
            "last_epoch": last_epoch
        }

        with open(self.config.checkpt_file(), 'wb') as out_weights_file:
            pickle.dump(parameters_to_save, out_weights_file, pickle.HIGHEST_PROTOCOL)

        with open(self.config.logs_file(), 'wb') as out_log_file:
            logs_to_save = {'last_tr_epoch': last_epoch,
                            'logs': self.logs,
                            'fully_trained': fully_trained}
            pickle.dump(logs_to_save, out_log_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)
        self.last_epoch = 1
        self.trained_flag = False
        self.logs = {}

    def restore_model(self) -> None:
        print("Restoring training logs from file %s." % self.config.logs_file())
        with open(self.config.logs_file(), 'rb') as in_file:
            full_logs = pickle.load(in_file)
            self.logs = full_logs['logs']
            self.trained_flag = full_logs['fully_trained']
        print("Restoring weights from file %s." % self.config.checkpt_file())
        with open(self.config.checkpt_file(), 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        shared_params = {k: self.params[k] for k in self.params if
                         k in data_to_load['params'] and self.params[k] == data_to_load['params'][k]}
        assert len(self.params) == len(data_to_load['params']) == len(shared_params)

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)

        self.last_epoch = data_to_load['last_epoch']


def cross_validation_GAT():
    hu_choices = [[20, 20, 10]]
    ah_choices = [[3, 3, 2]]
    aggr_choices = [MainGAT.master_node_aggregator, MainGAT.concat_feature_aggregator,MainGAT.average_feature_aggregator]
    include_weights = [True,False]
    pers_traits = [['NEO.NEOFAC_A']]
    batch_chocies = [2]
    for hu, ah, agg, iw, p_traits, batch_size in product(hu_choices, ah_choices, aggr_choices, include_weights,
                                                         pers_traits, batch_chocies):
        for eval_fold_out in range(5):
            for eval_fold_in in range(5):
                dict_param = {
                    'hidden_units': hu,
                    'attention_heads': ah,
                    'include_ew': iw,
                    'readout_aggregator': agg,
                    'load_specific_data': load_struct_data,
                    'pers_traits_selection': p_traits,
                    'batch_size': batch_size,
                    'eval_fold_in': eval_fold_in,
                    'eval_fold_out': eval_fold_out,
                    'k_outer': 5,
                    'k_inner': 5,
                    'nested_CV_level': 'outer'

                }
                model = CrossValidatedGAT(args=GAT_hyperparam_config(dict_param))
                model.train()
                model.test()


if __name__ == "__main__":
    cross_validation_GAT()
