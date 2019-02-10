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

    def sorted_stratification(self, unbalanced_subj, k_split, eval_fold, id_dict):
        sorted_id = [x[1] for x in sorted(id_dict.items(), key=operator.itemgetter(0))]
        target_trait = self.params['pers_traits_selection'][0]
        strat_split_file = os.path.join(self.config.proc_data_dir(), '_'.join(list(map(str, sorted_id))) + target_trait)
        if os.path.exists(strat_split_file + '.npy'):
            print('Reload the split of sorted stratification for the model %s' % self.config)
            stratified_subj = np.load(strat_split_file + '.npy')
            test_fold = stratified_subj[eval_fold]
            training = np.concatenate(np.delete(stratified_subj, obj=eval_fold, axis=0))
            return test_fold, training

        from random import randint
        sorted_subjs_by_score = sorted(unbalanced_subj, key=lambda x: self.data[x]['score'][0][0])
        stratified_subj = []
        for _ in range(k_split): stratified_subj.append([])

        for window_nr in range(len(sorted_subjs_by_score) // k_split):
            window = sorted_subjs_by_score[window_nr * k_split:(window_nr + 1) * k_split]
            assert len(window) == k_split
            scores_left_window = k_split
            for fold in range(k_split):
                random_index_window = randint(0, scores_left_window - 1)
                stratified_subj[fold].append(window[random_index_window])
                del window[random_index_window]
                scores_left_window -= 1

        dump_fold_rest = randint(0, k_split - 1)
        for rest in range(len(sorted_subjs_by_score) // k_split * k_split, len(sorted_subjs_by_score)):
            stratified_subj[dump_fold_rest].append(sorted_subjs_by_score[rest])

        stratified_subj = np.array(stratified_subj)
        np.save(strat_split_file, stratified_subj)

        test_fold = stratified_subj[eval_fold]
        training = np.concatenate(np.delete(stratified_subj, obj=eval_fold, axis=0))
        return test_fold, training

        # Split the entire dataset for a particular training and testing of the nested Cross Validation

    def split_nested_CV(self):
        eval_fold_in = self.params['eval_fold_in']  # the specific inner fold chosen for evaluation
        eval_fold_out = self.params['eval_fold_out']  # the specific outer fold chosen for evaluation
        k_outer = self.params['k_outer']  # the number of outer folds
        k_inner = self.params['k_inner']  # the number of inner folds
        # prepare the outer split
        stratif_identif = {'k_outer': k_outer,
                           'fold_usage': 'test',
                           'nested_CV_level': 'outer'}
        self.outer_test, out_training = self.sorted_stratification(unbalanced_subj=self.subjects, k_split=k_outer,
                                                                   eval_fold=eval_fold_out, id_dict=stratif_identif)
        stratif_identif = {'k_outer': k_outer,
                           'eval_fold_out': eval_fold_out,
                           'fold_usage': 'val',
                           'nested_CV_level': 'outer'}
        self.outer_validation, self.outer_train = self.sorted_stratification(unbalanced_subj=out_training,
                                                                             k_split=k_outer,
                                                                             eval_fold=-1, id_dict=stratif_identif)

        # prepare the inner split
        stratif_identif = {'k_outer': k_outer,
                           'k_inner': k_inner,
                           'eval_fold_out': eval_fold_out,
                           'fold_usage': 'test',
                           'nested_CV_level': 'inner'}
        self.inner_test, in_training = self.sorted_stratification(unbalanced_subj=out_training, k_split=k_inner,
                                                                  eval_fold=eval_fold_in, id_dict=stratif_identif)
        stratif_identif = {'k_outer': k_outer,
                           'k_inner': k_inner,
                           'eval_fold_in': eval_fold_in,
                           'eval_fold_out': eval_fold_out,
                           'fold_usage': 'val',
                           'nested_CV_level': 'inner'}

        self.inner_validation, self.inner_train = self.sorted_stratification(unbalanced_subj=in_training,
                                                                             k_split=k_inner,
                                                                             eval_fold=-1, id_dict=stratif_identif)

        assert set(self.inner_train).isdisjoint(set(self.inner_test))
        assert set(self.outer_train).isdisjoint(set(self.outer_test))

    def make_model(self):
        with tf.variable_scope('input'):
            batch_sz = None
            self.placeholders['ftr_in'] = tf.placeholder(dtype=tf.float32,
                                                         shape=[batch_sz, self.nb_nodes, self.ft_size])
            self.placeholders['bias_in'] = tf.placeholder(dtype=tf.float32,
                                                          shape=[batch_sz, self.nb_nodes, self.nb_nodes])
            self.placeholders['score_in'] = tf.placeholder(dtype=tf.float32,
                                                           shape=[batch_sz, self.outGAT_sz_target])
            self.placeholders['adj_in'] = tf.placeholder(dtype=tf.float32,
                                                         shape=[batch_sz, self.nb_nodes, self.nb_nodes])
            self.placeholders['is_train'] = tf.placeholder(dtype=tf.bool, shape=())

            # batch outputs inferred by GAT
            prediction, self.ops['unif_loss'], self.ops['excl_loss'] = \
                MainGAT.inference_keras(self, in_feat_vects=self.placeholders['ftr_in'],
                                        adj_mat=self.placeholders['adj_in'],
                                        bias_mat=self.placeholders['bias_in'],
                                        include_weights=self.params['include_ew'],
                                        hid_units=self.params['hidden_units'],
                                        n_heads=self.params['attention_heads'],
                                        target_score_type=self.outGAT_sz_target,
                                        aggregator=self.params['readout_aggregator'],
                                        is_train=self.placeholders['is_train'],
                                        residual=self.params['residual'],
                                        activation=self.params['non_linearity'],
                                        attn_drop=self.params['attn_drop'], )
            # per-batch loss
            self.ops['loss'] = tf.losses.mean_squared_error(labels=self.placeholders['score_in'],
                                                            predictions=prediction)

            # minibatch training op
            self.ops['train_op'] = super().training(loss=self.ops['loss'], u_loss=self.ops['unif_loss'],
                                                    e_loss=self.ops['excl_loss'], lr=self.params['learning_rate'],
                                                    l2_coef=self.params['l2_coefficient'])

    # run the list of Operation objects using the data of the subject with different dropout rate for training
    def feed_forward_op(self, ops, mini_batch, is_train):
        return self.sess.run(ops, feed_dict={self.placeholders['ftr_in']: mini_batch['ftr_in'],
                                             self.placeholders['bias_in']: mini_batch['bias_in'],
                                             self.placeholders['score_in']: mini_batch['score_in'],
                                             self.placeholders['adj_in']: mini_batch['adj_in'],
                                             self.placeholders['is_train']: is_train
                                             })

    def batch_generation(self, subj_keys):
        tr_iterations = len(subj_keys) // self.params['batch_size']
        batch_set = []
        for iteration in range(tr_iterations):
            batch_data = [self.data[key] for key in subj_keys[iteration * self.params['batch_size']:
                                                              (iteration + 1) * self.params['batch_size']]]
            mini_batch = {'ftr_in': [], 'bias_in': [], 'score_in': [], 'adj_in': []}
            for subj_data in batch_data:
                mini_batch['ftr_in'].append(subj_data['feat'])
                mini_batch['bias_in'].append(subj_data['bias'])
                mini_batch['score_in'].append(subj_data['score'])
                mini_batch['adj_in'].append(subj_data['adj'])
            batch_set.append(mini_batch)
            for key in mini_batch.keys():
                mini_batch[key] = np.array(mini_batch[key])

        return batch_set

    # run one batch of training: on training_set which comes from a k_split of the dataset (inner or outer)
    def run_epoch_training(self, training_set):
        # Train loop
        tr_size = len(training_set)
        # shuffle the training dataset
        shuf_subjs = shuffle_tr_data(training_set, tr_size)
        # generate batches
        tr_batch_set = self.batch_generation(shuf_subjs)
        # train using all batches
        for tr_batch in tr_batch_set:
            self.feed_forward_op(ops=[self.ops['train_op']], mini_batch=tr_batch, is_train=True)
        # calculate the loss on all the batches using the best refined weights/biases
        tr_avg_loss, avg_eloss, avg_uloss = 0.0, 0.0, 0.0
        for tr_batch in tr_batch_set:
            (tr_expl_loss, expl_uloss, expl_eloss) = self.feed_forward_op([self.ops['loss'], self.ops['unif_loss'],
                                                                           self.ops['excl_loss']],
                                                                          mini_batch=tr_batch,
                                                                          is_train=False)
            tr_avg_loss += tr_expl_loss
            avg_eloss += expl_eloss
            avg_uloss += expl_uloss

        return map(lambda x: x / len(tr_batch_set), [tr_avg_loss, avg_uloss, avg_eloss])

    def run_epoch_validation(self, validation_batch_set):
        # number of validation graph examples
        vl_size = len(validation_batch_set)
        vl_avg_loss = 0.0
        for vl_batch in validation_batch_set:
            (vl_expl_loss,) = self.feed_forward_op([self.ops['loss']], mini_batch=vl_batch, is_train=False)
            vl_avg_loss += vl_expl_loss
        vl_avg_loss /= vl_size

        return vl_avg_loss

    def train(self):
        if self.trained_flag:
            return
        # choose the correct training set of subjects
        if self.params['nested_CV_level'] == 'inner':
            training_set = self.inner_train
            validation_set = self.inner_validation
        else:
            training_set = self.outer_train
            validation_set = self.outer_validation

        tr_size = len(training_set)
        vl_size = len(validation_set)
        print('The training size is: %d and the validation %d' % (tr_size, vl_size))
        # the validation set is not shuffled, so batched generated just once
        vl_batch_set = self.batch_generation(validation_set)
        # record the minimum validation loss encountered until current epoch
        vlss_mn = np.inf
        # store the validation loss of previous epoch
        vlss_early_model = np.inf
        # record the number of consecutive epochs when the loss doesn't improve
        curr_step = 0

        for epoch in range(self.last_epoch, self.params['num_epochs']):
            total_time_start = time.time()
            epoch_tr_loss, epoch_uloss, epoch_eloss = self.run_epoch_training(training_set=training_set)
            epoch_val_loss = self.run_epoch_validation(validation_batch_set=vl_batch_set)
            self.logs[epoch] = {"tr_loss": epoch_tr_loss,
                                "val_loss": epoch_val_loss,
                                "u_loss": epoch_uloss,
                                "e_loss": epoch_eloss,
                                }
            epoch_time = time.time() - total_time_start
            print('Training: loss = %.5f | Val: loss = %.5f | '
                  'Unifrom loss: %f| Exclusive loss: %f | '
                  'Elapsed epoch time: %.5f' % (epoch_tr_loss, epoch_val_loss, epoch_uloss, epoch_eloss, epoch_time))
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
        #  generate batches for training
        self.params['batch_size'] = 1
        ts_batch_set = self.batch_generation(test_set)

        ts_avg_loss = 0.0
        for ts_batch in ts_batch_set:
            (ts_example_loss,) = self.feed_forward_op([self.ops['loss']], mini_batch=ts_batch, is_train=False)
            ts_avg_loss += ts_example_loss

        ts_avg_loss /= len(ts_batch_set)
        print('Test: loss = %.5f for the model %s' % (ts_avg_loss, self.config))
        self.sess.close()

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
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
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
    aggr_choices = [MainGAT.concat_feature_aggregator]
    include_weights = [True]
    pers_traits = [['NEO.NEOFAC_A'], ['NEO.NEOFAC_O'], ['NEO.NEOFAC_C'], ['NEO.NEOFAC_N'], ['NEO.NEOFAC_E']]
    batch_chocies = [2]
    load_choices = [load_struct_data]
    for load, hu, ah, agg, iw, p_traits, batch_size in product(load_choices, hu_choices, ah_choices, aggr_choices,
                                                               include_weights,
                                                               pers_traits, batch_chocies):
        for eval_in in range(5):
            dict_param = {
                'hidden_units': hu,
                'attention_heads': ah,
                'include_ew': iw,
                'readout_aggregator': agg,
                'load_specific_data': load,
                'pers_traits_selection': p_traits,
                'batch_size': batch_size,
                'eval_fold_in': eval_in,
                'eval_fold_out': 4,
                'k_outer': 5,
                'k_inner': 5,
                'nested_CV_level': 'outer'

            }
            model = CrossValidatedGAT(args=GAT_hyperparam_config(dict_param))
            model.train()
            model.test()


if __name__ == "__main__":
    cross_validation_GAT()
