#!/usr/bin/env/python

from typing import Tuple, List, Any, Sequence

import tensorflow as tf
import time
import os
import json
import numpy as np
import pickle
import random

from MainGAT import *


class ContractionMapGAT(MainGAT):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 10000,
            'patience': 100,
            'learning_rate': 0.0001,
            'l2_coefficient': 0.0005,
            'hidden_units': [16, 8],
            'attention_heads': [3, 2],
            'batch_size': 1,
            'readout_aggregator': super().concat_feature_aggregator,
            'edgeWeights_filter': interval_filter,
            'filter_limits': (183, 263857),
            'non_linearity': tf.nn.elu,
            'pers_traits_selection': ['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E'],
            'load_specific_data': load_struct_data,
            'resifual': False,
            'train_file': 'molecules_train.json',
            'valid_file': 'molecules_valid.json'
        }

    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        log_dir = args.get('--log_dir') or '.'
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        # Load data:

        # data for adjancency matrices, node feature vectors and personality scores for each study patient
        self.adj_matrices, self.graphs_features, self.score_train, self.score_test, self.score_val = params[
            'load_specific_data']
        # used in order to implement MASKED ATTENTION by discardining non-neighbours out of nhood hops
        self.biases = adj_to_bias(self.adj_matrices, [graph.shape[0] for graph in self.adj_matrices], nhood=1)
        # nr of nodes for each graph: it is shared among all examples due to the dataset
        self.nb_nodes = self.adj_matrices[0].shape[0]
        # the initial length F of each node feature vector: for every graph, node feat.vecs. have the same length
        self.ft_size = self.graphs_features.shape[-1]
        # how many of the big-five personality traits the model is targeting at once
        self.outGAT_sz_target = len(params['personality_traits_selection'])

        self.train_data = self.load_data(params['train_file'], is_training_data=True)
        self.valid_data = self.load_data(params['valid_file'], is_training_data=False)

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data:
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        return self.process_raw_graphs(data, is_training_data)

    def make_model(self):
        with tf.variable_scope("input"):
            self.placeholders['ftr_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.nb_nodes, self.ft_size))
            self.placeholders['bias_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.nb_nodes, self.nb_nodes))
            self.placeholders['score_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.outGAT_sz_target))
            self.placeholders['adj_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.nb_nodes, self.nb_nodes))
            self.placeholders['attn_drop'] = tf.placeholder(dtype=tf.float32, shape=())
            self.placeholders['ffd_drop'] = tf.placeholder(dtype=tf.float32, shape=())
            self.placeholders['is_train'] = tf.placeholder(dtype=tf.bool, shape=())

            prediction = MainGAT.inference(in_feat_vects=self.placeholders['ftr_in'],
                                           adj_mat=self.placeholders['adj_in'],
                                           bias_mat=self.placeholders['bias_in'],
                                           hid_units=self.params['hidden_units'],
                                           n_heads=self.params['attention_heads'],
                                           target_score_type=self.outGAT_sz_target,
                                           train_flag=self.placeholders['is_train'],
                                           aggregator=self.params['readout_aggregator'],
                                           include_weights=True if self.params[
                                                                       'edgeWeights_filter'] is not None else False,
                                           residual=self.params['residual'],
                                           activation=self.params['non_linearity'],
                                           attn_drop=self.placeholders['is_train'],
                                           ffd_drop=self.placeholders['ffd_drop'])

            self.ops['loss'] = tf.losses.mean_squared_error(labels=self.placeholders['score_in'],
                                                            predictions=prediction)

    def feedforward_op(self, op, index):
        return self.sess.run([op], feed_dict={self.placeholders['ftr_in']: self.ftr_in_tr[index:index + 1],
                                              self.placeholders['bias_in']: self.bias_in_tr[index:index + 1],
                                              self.placeholders['score_in']: self.score_in_tr[index:index + 1],
                                              self.placeholders['adj_in']: self.adj_in_tr[index:index + 1],
                                              self.placeholders['is_train']: True,
                                              self.placeholders['attn_drop']: 0.6,
                                              self.placeholders['ffd_drop']: 0.6})

    def make_train_step(self):

        # minibatch update operations
        zero_grads_ops, accum_ops, apply_ops = super().batch_training(loss=self.ops['loss'],
                                                                      lr=self.params['learning_rate'],
                                                                      l2_coef=self.params['l2_coefficient'])

        # number of training, validation, test graph examples
        tr_size, vl_size = len(self.score_train), len(self.score_val)
        print('The training dataset size is: %d, while for validation: %d' % (tr_size, vl_size))

        # reload the GAT model from the last checkpoint
        epoch_start = reload_GAT_model(model_GAT_choice=model_GAT_choice, sess=sess, saver=saver)
        # record the minimum validation loss encountered until current epoch
        vlss_mn = np.inf
        # record the number of consecutive epochs when the loss doesn't improve
        curr_step = 0

        # Train loop
        # nb_epochs - number of epochs for training: the number of iteration of gradient descent to optimize
        for epoch in range(epoch_start, self.params['epochs'] + 1):
            # number of iterations of the training set when batch-training
            tr_iterations = tr_size // self.params['batch_size']
            # Array for logging the loss
            tr_loss_log = np.zeros(tr_iterations)
            # shuffle the training dataset
            self.score_in_tr, self.ftr_in_tr, self.bias_in_tr, self.adj_in_tr = shuffle_tr_data(self.score_train,
                                                                                                self.graphs_features,
                                                                                                self.biases,
                                                                                                self.adj_matrices,
                                                                                                tr_size)
            for iteration in range(tr_iterations):
                # Make sure gradients are set to 0 before entering minibatch loop
                self.sess.run(zero_grads_ops)
                # Loop over minibatches and execute accumulate-gradient operation
                for batch_step in range(self.params['batch_size']):
                    index = batch_step + iteration * self.params['batch_size']
                    self.feedforward_op(accum_ops, index)
                # Done looping over minibatches. Now apply gradients.
                self.sess.run(apply_ops)
                # Calculate the validation loss after every single batch training
                for batch_step in range(self.params['batch_size']):
                    index = batch_step + iteration * self.params['batch_size']
                    (tr_example_loss,) = self.feedforward_op(self.ops['loss'], index)
                    tr_loss_log[iteration] += tr_example_loss
                tr_loss_log[iteration] /= self.params['batch_size']

            vl_avg_loss = 0
            for vl_step in range(tr_size, tr_size + vl_size):
                (vl_example_loss,) = self.sess.run([self.ops['loss']],
                                                   feed_dict={self.placeholders['ftr_in']: self.graphs_features[
                                                                                           vl_step:vl_step + 1],
                                                              self.placeholders['bias_in']: self.biases[
                                                                                            vl_step:vl_step + 1],
                                                              self.placeholders['score_in']: self.score_val[
                                                                                             vl_step - tr_size:vl_step - tr_size + 1],
                                                              self.placeholders['adj_in']: self.adj_matrices[
                                                                                           vl_step:vl_step + 1],
                                                              self.placeholders['is_train']: False,
                                                              self.placeholders['attn_drop']: 0.0,
                                                              self.placeholders['ffd_drop']: 0.0})
                vl_avg_loss += vl_example_loss
            vl_avg_loss /= vl_size

            tr_avg_loss = np.sum(tr_loss_log) / (tr_size // self.params['batch_size'])
            print('Training: loss = %.5f | Val: loss = %.5f' % (tr_avg_loss, vl_avg_loss))


    def run_epoch(self, epoch_name: str, data, is_training: bool):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])

        loss = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = None
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params[
                    'out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_ops]
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracies) = (result[0], result[1])
            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)

            print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
                                                                               step,
                                                                               num_graphs,
                                                                               loss / processed_graphs),
                  end='\r')

        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, accuracies, error_ratios, instance_per_sec

    def train(self):
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                _, valid_accs, _, _ = self.run_epoch("Resumed (validation)", self.valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (float("+inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_accs, train_errs, train_speed = self.run_epoch("epoch %i (training)" % epoch,
                                                                                 self.train_data, True)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                print("\r\x1b[K Train: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (train_loss,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        train_speed))
                valid_loss, valid_accs, valid_errs, valid_speed = self.run_epoch("epoch %i (validation)" % epoch,
                                                                                 self.valid_data, False)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print("\r\x1b[K Valid: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (valid_loss,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        valid_speed))

                epoch_time = time.time() - total_time_start
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': (train_loss, train_accs.tolist(), train_errs.tolist(), train_speed),
                    'valid_results': (valid_loss, valid_accs.tolist(), valid_errs.tolist(), valid_speed),
                }
                log_to_save.append(log_entry)
                with open(self.log_file, 'w') as f:
                    json.dump(log_to_save, f, indent=4)

                val_acc = np.sum(valid_accs)  # type: float
                if val_acc < best_val_acc:
                    self.save_model(self.best_model_file)
                    print("  (Best epoch so far, cum. val. acc decreased to %.5f from %.5f. Saving to '%s')" % (
                        val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params[
                        'patience'])
                    break

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "params": self.params,
            "weights": weights_to_save
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different task_ids:
            if par not in ['task_ids', 'num_epochs']:
                assert par_value == data_to_load['params'][par]

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
