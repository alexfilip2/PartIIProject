import time
import pickle
import multiprocessing
from gat_impl.MainGAT import *
import os.path
from gat_impl.HyperparametersGAT import HyperparametersGAT
import math


class CrossValidatedGAT(MainGAT):
    @classmethod
    def default_params(cls):
        return HyperparametersGAT()

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
        # Initialize parameters of the data
        self.true_scores = []
        self.tr_size = self.vl_size = self.ts_size = 0
        # Initialize the model skeleton, the computation graph and the parameters
        self.training_init_op = self.validation_init_op = self.testing_init_op = None
        self.iterator = None
        self.last_epoch = 1
        self.trained_flag = False
        self.logs = {}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = multiprocessing.cpu_count()
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.params['random_seed'])
            self.placeholders = {}
            self.ops = {}

    # Load the entire dataset structural or functional that will be used
    def load_pipeline_data(self, data, train_subj, val_subj, test_subj):
        # true predictions for the test set
        self.true_scores = [data[subj]['score_in'][0] for subj in test_subj]
        # nr of nodes of each graph
        nb_nodes = data[train_subj[0]]['adj_in'].shape[-1]
        # the initial dimension F of each node's feature vector
        ft_size = data[train_subj[0]]['ftr_in'].shape[-1]

        def format_for_pipeline(subj_keys):
            data_sz = len(subj_keys)
            entire_data = {'ftr_in': np.empty(shape=(data_sz, nb_nodes, ft_size), dtype=np.float32),
                           'bias_in': np.empty(shape=(data_sz, nb_nodes, nb_nodes), dtype=np.float32),
                           'adj_in': np.empty(shape=(data_sz, nb_nodes, nb_nodes), dtype=np.float32),
                           'score_in': np.empty(shape=(data_sz, self.params['target_score_type']), dtype=np.float32)}

            for expl_index, s_key in enumerate(subj_keys):
                for input_type in data[s_key].keys():
                    entire_data[input_type][expl_index] = data[s_key][input_type]

            return entire_data

        # choose the suitable dataset for the CV level and format it for use with a tf Dataset
        data_sets = list(map(lambda subj_set: format_for_pipeline(subj_set), [train_subj, val_subj, test_subj]))
        tr_slices, vl_slices, ts_slices = map(lambda data_set: (data_set['ftr_in'],
                                                                data_set['bias_in'],
                                                                data_set['adj_in'],
                                                                data_set['score_in']), data_sets)
        # Size of the datasets used for this GAT model
        self.tr_size, self.vl_size, self.ts_size = len(train_subj), len(val_subj), len(test_subj)
        print('The training size is %d, the validation one: %d and the test one: %d' % (
            self.tr_size, self.vl_size, self.ts_size))

        # allow for dynamically changing of the batch size (supported by the underlying archit.) and Dropout rate
        self.placeholders['is_train'] = tf.placeholder(dtype=tf.bool, shape=())
        self.placeholders['batch_size'] = tf.placeholder(dtype=tf.int64, shape=())

        # create the Dataset objects pipeline the individual datasets, shuffle the train set then generate batched
        tr_dataset = tf.data.Dataset.from_tensor_slices(tr_slices).shuffle(buffer_size=1000).batch(
            self.placeholders['batch_size']).repeat()
        vl_dataset = tf.data.Dataset.from_tensor_slices(vl_slices).batch(self.vl_size).repeat()
        ts_dataset = tf.data.Dataset.from_tensor_slices(ts_slices).batch(self.tr_size)

        # create an iterator for the datasets which will extract a batch at a time
        iterator = tf.data.Iterator.from_structure(tr_dataset.output_types, tr_dataset.output_shapes)
        # operations to swap between datsets
        self.training_init_op = iterator.make_initializer(tr_dataset)
        self.validation_init_op = iterator.make_initializer(vl_dataset)
        self.testing_init_op = iterator.make_initializer(ts_dataset)
        self.iterator = iterator

    def build(self):
        with tf.variable_scope('input'):
            # right order to unpack is ftr_in, bias_in, adj_in, score_in
            batch_node_features, batch_bias_mats, batch_adj_mats, batch_scores = self.iterator.get_next()
            feed_data = {'batch_node_features': batch_node_features,
                         'batch_bias_mats': batch_bias_mats,
                         'batch_adj_mats': batch_adj_mats,
                         'is_train': self.placeholders['is_train']}
            # parameters and inputs for building the graph
            inference_args = {**feed_data, **self.params}

            # batch outputs inferred by GAT, losses for uniformity and exclusivity regulations
            self.ops['prediction'], self.ops['u_loss'], self.ops['e_loss'] = MainGAT.inference_keras(self,
                                                                                                     **inference_args)

            # per-batch MSE prediction loss
            self.ops['loss'] = tf.losses.mean_squared_error(labels=batch_scores, predictions=self.ops['prediction'])
            # update the mean and variance of the Batch Normalization at each mini-batch trining step
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # training operation
                train_args = {**self.ops, **self.params}
                self.ops['train_op'] = super().training(**train_args)

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
            self.last_epoch = data_to_load['last_epoch']

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

    def train(self):
        # Restore/initialize variables:
        if os.path.exists(self.config.checkpt_file()) and os.path.exists(self.config.logs_file()):
            self.restore_model()
        else:
            self.initialize_model()
        if self.trained_flag:
            return

        # keep track of best val loss and the last k training losses for early-stopping the training
        best_vl_loss = np.inf
        tr_k_logs = np.zeros(self.params['k_strip_epochs'])

        # run the training-validation cycle
        for epoch in range(self.last_epoch, self.params['num_epochs']):
            total_time_start = time.time()

            # fill the TensorFlow intializable Dataset with the training data
            self.sess.run(self.training_init_op, feed_dict={self.placeholders['batch_size']: self.params['batch_size']})

            # perform mini-batch training
            for i in range(math.ceil(self.tr_size / self.params['batch_size'])):
                self.sess.run([self.ops['train_op']], feed_dict={self.placeholders['is_train']: True})
            # compute the training loss feeding the whole dataset as a batch
            self.sess.run(self.training_init_op, feed_dict={self.placeholders['batch_size']: self.tr_size})
            epoch_tr_loss, epoch_uloss, epoch_eloss = self.sess.run([self.ops['loss'], self.ops['u_loss'],
                                                                     self.ops['e_loss']],
                                                                    feed_dict={self.placeholders['is_train']: False})
            # average the total epoch losses by batch number
            epoch_uloss /= self.tr_size
            epoch_eloss /= self.tr_size

            # fill the TensorFlow intializable Dataset with the validation data, feed it into in one batch
            self.sess.run(self.validation_init_op)
            (epoch_val_loss,) = self.sess.run([self.ops['loss']], feed_dict={self.placeholders['is_train']: False})
            # log the loss values so far
            self.logs[epoch] = {"tr_loss": epoch_tr_loss, "val_loss": epoch_val_loss}

            # eraly stop the training based on generalization loss vs training progress
            def early_stopping(tr_k_logs, epoch_tr_loss, best_vl_loss, epoch_val_loss):
                # pop last tr loss and push the current one
                tr_k_logs[:-1] = tr_k_logs[1:]
                tr_k_logs[-1] = epoch_tr_loss
                if np.min(tr_k_logs) > 0.0:
                    best_k_tr_loss = np.min(tr_k_logs)
                    last_k_tr_loss = np.sum(tr_k_logs)
                    gl = (epoch_val_loss / best_vl_loss - 1.0) * 100.0
                    pk = (last_k_tr_loss / (self.params['k_strip_epochs'] * best_k_tr_loss) - 1.0) * 1000.0
                    print('PQ ratio for epoch is %.6f and the training progress %.5f' % (gl / pk, pk))
                    if gl / pk >= self.params['gl_tr_prog_threshold'] and pk <= self.params['enough_train_prog']:
                        return True
                    if pk <= self.params['no_train_prog']:
                        return True

                return False

            # update the generalization loss
            best_vl_loss = min(best_vl_loss, epoch_val_loss)
            if early_stopping(tr_k_logs, epoch_tr_loss, best_vl_loss, epoch_val_loss):
                self.save_model(best_vl_loss=best_vl_loss, epoch_val_loss=epoch_val_loss, last_epoch=epoch,
                                fully_trained=True)
                break
            if epoch % self.params['CHECKPT_PERIOD'] == 0:
                self.save_model(best_vl_loss=best_vl_loss, epoch_val_loss=epoch_val_loss, last_epoch=epoch + 1,
                                fully_trained=False)
            epoch_time = time.time() - total_time_start
            print('Training: loss = %.5f | Val: loss = %.5f | Unifrom loss: %f| Exclusive loss: %f | '
                  'Elapsed epoch time: %.5f' % (epoch_tr_loss, epoch_val_loss, epoch_uloss, epoch_eloss, epoch_time))

    def test(self):
        # fill the TensorFlow intializable Dataset with the testing data, feed it into in one batch
        self.sess.run(self.testing_init_op)
        (ts_avg_loss, results) = self.sess.run([self.ops['loss'], self.ops['prediction']],
                                               feed_dict={self.placeholders['is_train']: False})
        print('Test: loss = %.5f for the model %s' % (ts_avg_loss, self.config))
        with open(self.config.results_file(), 'wb') as out_result_file:
            results = {'predictions': np.array(zip(results.flatten(), self.true_scores)), 'test_loss': ts_avg_loss}
            pickle.dump(results, out_result_file, pickle.HIGHEST_PROTOCOL)
        self.sess.close()

    def save_model(self, best_vl_loss, epoch_val_loss, last_epoch: int, fully_trained: bool = False) -> None:
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
        if fully_trained:
            print("Training progress after %d epochs saved in path: %s" % (last_epoch, self.config.checkpt_file()))
            print('Early stop! Min loss: ', best_vl_loss)
            print('Early stop model validation loss: ', epoch_val_loss)
        else:
            print("Training progress after %d epochs saved in path: %s" % (last_epoch, self.config.checkpt_file()))
