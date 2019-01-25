import tensorflow as tf
import time
import os
import json
import numpy as np
import pickle
import random

from MainGAT import *


class CrossValidatedGAT(MainGAT):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 10000,
            'patience': 100,
            'learning_rate': 0.0001,
            'l2_coefficient': 0.0005,
            'hidden_units': [16, 8],
            'attention_heads': [3, 2],
            'batch_size': 2,
            'readout_aggregator': super().concat_feature_aggregator,
            'include_ew': True,
            'edgeWeights_filter': interval_filter,
            'filter_limits': (183, 263857),
            'non_linearity': tf.nn.elu,
            'pers_traits_selection': ['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E'],
            'load_specific_data': load_struct_data,
            'residual': False,
            'random_seed': 123,
            'train_file': 'molecules_train.json',
            'valid_file': 'molecules_valid.json'
        }

    def __init__(self, args):
        self.args = args
        params = self.default_params()
        self.params = params
        # Load data:
        # data for adjancency matrices, node feature vectors, biases for masked attention and personality scores
        self.data, self.subjects = self.params['load_specific_data']()
        # nr of nodes for each graph: it is shared among all examples due to the dataset
        self.nb_nodes = self.data[self.subjects[0]]['adj'].shape[-1]
        # the initial length F of each node feature vector: for every graph, node feat.vecs. have the same length
        self.ft_size = self.data[self.subjects[0]]['feat'].shape[-1]
        # how many of the big-five personality traits the model is targeting at once
        self.outGAT_sz_target = len(self.params['pers_traits_selection'])

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
            # Restore/initialize variables:
            if args is not None:
                restore_file = args.get('--restore')
            else:
                restore_file = None

            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

            self.train()

    def load_data(self, file_name):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        return data

    def make_model(self):
        with tf.variable_scope('input'):
            self.placeholders['ftr_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.nb_nodes, self.ft_size))
            self.placeholders['bias_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.nb_nodes, self.nb_nodes))
            self.placeholders['score_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.outGAT_sz_target))
            self.placeholders['adj_in'] = tf.placeholder(dtype=tf.float32, shape=(1, self.nb_nodes, self.nb_nodes))
            self.placeholders['attn_drop'] = tf.placeholder(dtype=tf.float32, shape=())
            self.placeholders['ffd_drop'] = tf.placeholder(dtype=tf.float32, shape=())
            self.placeholders['is_train'] = tf.placeholder(dtype=tf.bool, shape=())

            prediction, self.ops['unif_loss'], self.ops['excl_loss'] = MainGAT.inference(self,
                                                                                         in_feat_vects=
                                                                                         self.placeholders[
                                                                                             'ftr_in'],
                                                                                         adj_mat=self.placeholders[
                                                                                             'adj_in'],
                                                                                         bias_mat=self.placeholders[
                                                                                             'bias_in'],
                                                                                         hid_units=self.params[
                                                                                             'hidden_units'],
                                                                                         n_heads=self.params[
                                                                                             'attention_heads'],
                                                                                         target_score_type=self.outGAT_sz_target,
                                                                                         train_flag=
                                                                                         self.placeholders[
                                                                                             'is_train'],
                                                                                         aggregator=self.params[
                                                                                             'readout_aggregator'],
                                                                                         include_weights=
                                                                                         self.params[
                                                                                             'include_ew'],
                                                                                         residual=self.params[
                                                                                             'residual'],
                                                                                         activation=self.params[
                                                                                             'non_linearity'],
                                                                                         attn_drop=
                                                                                         self.placeholders[
                                                                                             'attn_drop'],
                                                                                         ffd_drop=self.placeholders[
                                                                                             'ffd_drop'])

            self.ops['loss'] = tf.losses.mean_squared_error(labels=self.placeholders['score_in'],
                                                            predictions=prediction)
            # minibatch update operations
            self.ops['zero_grads_ops'], self.ops['accum_ops'], self.ops['apply_ops'] = super().batch_training(
                loss=self.ops['loss'],
                u_loss=self.ops['unif_loss'],
                e_loss=self.ops['excl_loss'],
                lr=self.params['learning_rate'],
                l2_coef=self.params['l2_coefficient'])

    def feed_forward_op(self, op, index, examples_sbj, is_train, attn_drop, ffd_drop):
        subj_data = self.data[examples_sbj[index]]
        return self.sess.run([op], feed_dict={self.placeholders['ftr_in']: subj_data['feat'],
                                              self.placeholders['bias_in']: subj_data['bias'],
                                              self.placeholders['score_in']: subj_data['score'],
                                              self.placeholders['adj_in']: subj_data['adj'],
                                              self.placeholders['is_train']: is_train,
                                              self.placeholders['attn_drop']: attn_drop,
                                              self.placeholders['ffd_drop']: ffd_drop})

    def batch_train_step(self, iteration, examples_sbj):

        batch_avg_loss = 0.0
        # Make sure gradients are set to 0 before entering minibatch loop
        self.sess.run(self.ops['zero_grads_ops'])
        # Loop over minibatches and execute accumulate-gradient operation
        for batch_step in range(self.params['batch_size']):
            index = batch_step + iteration * self.params['batch_size']
            self.feed_forward_op(self.ops['accum_ops'], index, examples_sbj, is_train=True, attn_drop=0.6, ffd_drop=0.6)
        # Done looping over minibatches. Now apply gradients.
        self.sess.run(self.ops['apply_ops'])
        # Calculate the validation loss after every single batch training
        for batch_step in range(self.params['batch_size']):
            index = batch_step + iteration * self.params['batch_size']
            (tr_example_loss,) = self.feed_forward_op(self.ops['loss'], index, examples_sbj, is_train=True,
                                                      attn_drop=0.6,
                                                      ffd_drop=0.6)
            batch_avg_loss += tr_example_loss

        batch_avg_loss /= self.params['batch_size']

        return batch_avg_loss

    def run_epoch_training(self):
        # Train loop
        # number of training, validation, test graph examples
        split_sz = len(self.subjects) // 10
        tr_size = split_sz * 8
        # number of iterations of the training set when batch-training
        tr_iterations = tr_size // self.params['batch_size']
        # Array for logging the loss
        tr_loss_log = np.zeros(tr_iterations)
        # shuffle the training dataset
        shuf_subjs = shuffle_tr_data(self.subjects, tr_size)
        for iteration in range(tr_iterations):
            tr_loss_log[iteration] = self.batch_train_step(iteration=iteration, examples_sbj=shuf_subjs)

        return np.sum(tr_loss_log) / tr_iterations

    def run_epoch_validation(self):
        # number of training, validation, test graph examples
        split_sz = len(self.subjects) // 10
        tr_size, vl_size = split_sz * 8, split_sz
        vl_avg_loss = 0
        for vl_step in range(tr_size, tr_size + vl_size):
            (vl_example_loss,) = self.feed_forward_op(self.ops['loss'], index=vl_step, examples_sbj=self.subjects,
                                                   is_train=False, attn_drop=0.0, ffd_drop=0.0)
            vl_avg_loss += vl_example_loss

        vl_avg_loss /= vl_size
        return vl_avg_loss

    def train(self):
        for epoch in range((self.params['num_epochs'])):
            total_time_start = time.time()
            epoch_tr_loss = self.run_epoch_training()
            epoch_val_loss = self.run_epoch_validation()
            epoch_time = time.time() - total_time_start
            print('Training: loss = %.5f | Val: loss = %.5f | Elapsed time %.5f' % (
                epoch_tr_loss, epoch_val_loss, epoch_time))

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


if __name__ == "__main__":
    model = CrossValidatedGAT(args=None)
