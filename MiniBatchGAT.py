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
        sorted_subjs_by_score = sorted(unbalanced_subj, key=lambda x: self.data[x]['score'][0])
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
        self.outer, self.inner = {}, {}
        eval_fold_in = self.params['eval_fold_in']  # the specific inner fold chosen for evaluation
        eval_fold_out = self.params['eval_fold_out']  # the specific outer fold chosen for evaluation
        k_outer = self.params['k_outer']  # the number of outer folds
        k_inner = self.params['k_inner']  # the number of inner folds
        # prepare the outer split
        stratif_identif = {'k_outer': k_outer,
                           'fold_usage': 'test',
                           'nested_CV_level': 'outer'}
        self.outer['test'], out_training = self.sorted_stratification(unbalanced_subj=self.subjects, k_split=k_outer,
                                                                      eval_fold=eval_fold_out, id_dict=stratif_identif)
        stratif_identif = {'k_outer': k_outer,
                           'eval_fold_out': eval_fold_out,
                           'fold_usage': 'val',
                           'nested_CV_level': 'outer'}
        self.outer['validation'], self.outer['train'] = self.sorted_stratification(unbalanced_subj=out_training,
                                                                                   eval_fold=-1,
                                                                                   k_split=k_outer,
                                                                                   id_dict=stratif_identif)
        # prepare the inner split
        stratif_identif = {'k_outer': k_outer,
                           'k_inner': k_inner,
                           'eval_fold_out': eval_fold_out,
                           'fold_usage': 'test',
                           'nested_CV_level': 'inner'}
        self.inner['test'], in_training = self.sorted_stratification(unbalanced_subj=out_training, k_split=k_inner,
                                                                     eval_fold=eval_fold_in, id_dict=stratif_identif)
        stratif_identif = {'k_outer': k_outer,
                           'k_inner': k_inner,
                           'eval_fold_in': eval_fold_in,
                           'eval_fold_out': eval_fold_out,
                           'fold_usage': 'val',
                           'nested_CV_level': 'inner'}

        self.inner['validation'], self.inner['train'] = self.sorted_stratification(unbalanced_subj=in_training,
                                                                                   eval_fold=-1,
                                                                                   k_split=k_inner,
                                                                                   id_dict=stratif_identif)
        assert set(self.inner['train']).isdisjoint(set(self.inner['test']))
        assert set(self.inner['train']).isdisjoint(set(self.inner['validation']))
        assert set(self.outer['train']).isdisjoint(set(self.outer['test']))
        assert set(self.outer['train']).isdisjoint(set(self.outer['validation']))

    def format_data_pipeline(self, subj_keys):
        data_sz = len(subj_keys)
        whole_data = {'ftr_in': np.empty(shape=(data_sz, self.nb_nodes, self.ft_size), dtype=np.float32),
                      'bias_in': np.empty(shape=(data_sz, self.nb_nodes, self.nb_nodes), dtype=np.float32),
                      'score_in': np.empty(shape=(data_sz, self.outGAT_sz_target), dtype=np.float32),
                      'adj_in': np.empty(shape=(data_sz, self.nb_nodes, self.nb_nodes), dtype=np.float32)}
        for expl_index, s_key in enumerate(subj_keys):
            whole_data['ftr_in'][expl_index] = self.data[s_key]['feat']
            whole_data['bias_in'][expl_index] = self.data[s_key]['bias']
            whole_data['score_in'][expl_index] = self.data[s_key]['score']
            whole_data['adj_in'][expl_index] = self.data[s_key]['adj']

        return whole_data

    def make_model(self):
        with tf.variable_scope('input'):
            # choose the suitable dataset for the CV level and format it for use with a tf Dataset
            data = self.inner if self.params['nested_CV_level'] == 'inner' else self.outer
            training_set = self.format_data_pipeline(data['train'])
            validation_set = self.format_data_pipeline(data['validation'])
            test_set = self.format_data_pipeline(data['test'])
            self.tr_size, self.vl_size, self.ts_size = len(data['train']), len(data['validation']), len(data['test'])

            # allow for dynamically changing of the batch size (supported by the underlying archit.) and Dropout rate
            self.placeholders['is_train'] = tf.placeholder(dtype=tf.bool, shape=())
            self.placeholders['batch_size'] = tf.placeholder(tf.int64)
            # create the Dataset objects pipeline the individual datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((training_set['ftr_in'],
                                                                training_set['bias_in'],
                                                                training_set['adj_in'],
                                                                training_set['score_in']))
            # shuffle the train set then generate batched
            train_dataset = train_dataset.shuffle(buffer_size=1000).batch(self.placeholders['batch_size']).repeat()
            validation_dataset = tf.data.Dataset.from_tensor_slices((validation_set['ftr_in'],
                                                                     validation_set['bias_in'],
                                                                     validation_set['adj_in'],
                                                                     validation_set['score_in']))
            validation_dataset = validation_dataset.batch(self.placeholders['batch_size']).repeat()
            test_dataset = tf.data.Dataset.from_tensor_slices((test_set['ftr_in'],
                                                               test_set['bias_in'],
                                                               test_set['adj_in'],
                                                               test_set['score_in']))
            test_dataset = test_dataset.batch(self.placeholders['batch_size'])

            # create an iterator for the datasets which will extract a batch at a time
            iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            feats, biases, adjs, scores = iterator.get_next()
            # operations to swap between datsets
            self.training_init_op = iterator.make_initializer(train_dataset)
            self.validation_init_op = iterator.make_initializer(validation_dataset)
            self.testing_init_op = iterator.make_initializer(test_dataset)

            # batch outputs inferred by GAT
            prediction, unif_loss, excl_loss = MainGAT.inference_keras(self, in_feat_vects=feats,
                                                                       adj_mat=adjs,
                                                                       bias_mat=biases,
                                                                       include_weights=self.params['include_ew'],
                                                                       hid_units=self.params['hidden_units'],
                                                                       n_heads=self.params['attention_heads'],
                                                                       target_score_type=self.outGAT_sz_target,
                                                                       aggregator=self.params['readout_aggregator'],
                                                                       is_train=self.placeholders['is_train'],
                                                                       residual=self.params['residual'],
                                                                       activation=self.params['non_linearity'],
                                                                       attn_drop=self.params['attn_drop'])
            # losses for uniformity and exclusivity regulations
            self.ops['unif_loss'], self.ops['excl_loss'] = unif_loss, excl_loss
            # per-batch MSE prediction loss
            self.ops['loss'] = tf.losses.mean_squared_error(labels=scores, predictions=prediction)

            # minibatch training op
            self.ops['train_op'] = super().training(loss=self.ops['loss'],
                                                    u_loss=self.ops['unif_loss'],
                                                    e_loss=self.ops['excl_loss'],
                                                    lr=self.params['learning_rate'],
                                                    l2_coef=self.params['l2_coefficient'])

    def train(self):
        if self.trained_flag:
            return
        best_vl_loss = np.inf
        # store the validation loss of previous epoch
        prev_epoch_vl_loss = np.inf
        # record the number of consecutive epochs when the loss doesn't improve
        curr_step = 0
        n_batches = math.ceil(self.tr_size / self.params['batch_size'])

        # run the training-validation cycle
        for epoch in range(self.last_epoch, self.params['num_epochs']):
            total_time_start = time.time()
            # fill the TensorFlow intializable Dataset with the training data
            self.sess.run(self.training_init_op, feed_dict={self.placeholders['batch_size']: self.params['batch_size']})
            # perform mini-batch training
            for _ in range(n_batches):
                self.sess.run([self.ops['train_op']], feed_dict={self.placeholders['is_train']: True})
            # compute the training loss feeding the whole dataset as a batch
            self.sess.run(self.training_init_op, feed_dict={self.placeholders['batch_size']: self.tr_size})
            epoch_tr_loss, epoch_uloss, epoch_eloss = self.sess.run([self.ops['loss'],
                                                                     self.ops['unif_loss'],
                                                                     self.ops['excl_loss']],
                                                                    feed_dict={self.placeholders['is_train']: True})
            # average the total epoch losses by batch number
            epoch_uloss /= self.tr_size
            epoch_eloss /= self.tr_size
            # fill the TensorFlow intializable Dataset with the validation data, feed it into in one batch
            self.sess.run(self.validation_init_op, feed_dict={self.placeholders['batch_size']: self.vl_size})
            (epoch_val_loss,) = self.sess.run([self.ops['loss']], feed_dict={self.placeholders['is_train']: False})
            # log the loss values so far
            self.logs[epoch] = {"tr_loss": epoch_tr_loss,
                                "val_loss": epoch_val_loss,
                                "u_loss": epoch_uloss,
                                "e_loss": epoch_eloss,
                                }

            epoch_time = time.time() - total_time_start
            print('Training: loss = %.5f | Val: loss = %.5f | Unifrom loss: %f| Exclusive loss: %f | '
                  'Elapsed epoch time: %.5f' % (epoch_tr_loss, epoch_val_loss, epoch_uloss, epoch_eloss, epoch_time))
            if epoch % self.params['CHECKPT_PERIOD'] == 0:
                self.save_model(last_epoch=epoch + 1, fully_trained=False)
                print("Training progress after %d epochs saved in path: %s" % (epoch, self.config.checkpt_file()))

            # wait for the validation loss to settle before the specified number of training iterations
            if epoch_val_loss <= best_vl_loss:
                best_vl_loss = epoch_val_loss
                curr_step = 0
            else:
                curr_step += 1
            # store the prev epoch val loss
            prev_epoch_vl_loss = epoch_val_loss
            # check if the settleing reached the patience threshold
            if curr_step == self.params['patience'] and epoch >= 100:
                self.save_model(last_epoch=epoch, fully_trained=True)
                print("Training progress after %d epochs saved in path: %s" % (epoch, self.config.checkpt_file()))
                print('Early stop! Min loss: ', best_vl_loss)
                print('Early stop model validation loss: ', prev_epoch_vl_loss)
                break

    def test(self):
        # fill the TensorFlow intializable Dataset with the testing data, feed it into in one batch
        self.sess.run(self.testing_init_op, feed_dict={self.placeholders['batch_size']: self.ts_size})
        (ts_avg_loss,) = self.sess.run([self.ops['loss']], feed_dict={self.placeholders['is_train']: False})
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
    aggr_choices = [MainGAT.master_node_aggregator]
    include_weights = [True]
    pers_traits = [['NEO.NEOFAC_A'], ['NEO.NEOFAC_O'], ['NEO.NEOFAC_C'], ['NEO.NEOFAC_N'], ['NEO.NEOFAC_E']]
    batch_chocies = [2]
    load_choices = [load_struct_data]
    for load, hu, ah, agg, iw, p_traits, batch_size in product(load_choices, hu_choices, ah_choices, aggr_choices,
                                                               include_weights,
                                                               pers_traits, batch_chocies):
        for eval_out in range(5):
            dict_param = {
                'hidden_units': hu,
                'attention_heads': ah,
                'include_ew': iw,
                'readout_aggregator': agg,
                'load_specific_data': load,
                'pers_traits_selection': p_traits,
                'batch_size': batch_size,
                'eval_fold_in': 4,
                'eval_fold_out': eval_out,
                'k_outer': 5,
                'k_inner': 5,
                'nested_CV_level': 'outer'

            }
            model = CrossValidatedGAT(args=GAT_hyperparam_config(dict_param))
            model.train()
            model.test()


if __name__ == "__main__":
    cross_validation_GAT()
