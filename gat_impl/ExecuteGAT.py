import time
import pickle
import multiprocessing
from gat_impl.MainGAT import *
import os.path
from gat_impl.HyperparametersGAT import HyperparametersGAT
import math
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.backend import clear_session
import matplotlib.pyplot as plt

best_vl_loss = np.inf
total_time_start = 0.0


class GAT_Model(MainGAT):
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
        self.model = None
        self.already_loaded = False
        self.is_built = False

    # Load the entire dataset structural or functional that will be used
    def load_data(self, data, dataset_subjs):
        if not self.already_loaded:
            # the entire dataset
            self.data = data
            # nr of nodes of each graph
            self.N = data[dataset_subjs[0]]['adj_in'].shape[-1]
            # the initial dimension F of each node's feature vector
            self.F = data[dataset_subjs[0]]['ftr_in'].shape[-1]
            # mark the loading of fulld data into the model
            self.already_loaded = True

        def format_for_pipeline(subj_keys):
            data_sz = len(subj_keys)
            entire_data = {'ftr_in': np.empty(shape=(data_sz, self.N, self.F), dtype=np.float32),
                           'bias_in': np.empty(shape=(data_sz, self.N, self.N), dtype=np.float32),
                           'adj_in': np.empty(shape=(data_sz, self.N, self.N), dtype=np.float32),
                           'score_in': np.empty(shape=(data_sz, self.params['target_score_type']), dtype=np.float32)}

            for expl_index, s_key in enumerate(subj_keys):
                for input_type in data[s_key].keys():
                    entire_data[input_type][expl_index] = self.data[s_key][input_type]

            return entire_data

        # choose the suitable dataset for the CV level and format it for use with a tf Dataset
        zipped_data = format_for_pipeline(dataset_subjs)
        keras_formated = (zipped_data['ftr_in'], zipped_data['adj_in'], zipped_data['bias_in'], zipped_data['score_in'])
        return keras_formated

    def build(self):
        if self.is_built:
            return
            # right order to unpack is ftr_in, bias_in, adj_in, score_in
        feed_data = {'dim_nodes': self.N,
                     'dim_feats': self.F}
        # parameters and inputs for building the graph
        inference_args = {**feed_data, **self.params}

        # batch outputs inferred by GAT, losses for uniformity and exclusivity regulations
        self.model, mode_loss = MainGAT.inference_keras(self, **inference_args)

        optimizer = Adam(lr=self.params['learning_rate'])
        self.model.compile(optimizer=optimizer, loss=mode_loss)
        self.is_built = True

    def fit(self, data, train_subj, val_subj):

        tr_feats, tr_adjs, tr_biases, tr_scores = self.load_data(data=data, dataset_subjs=train_subj)
        vl_feats, vl_adjs, vl_biases, vl_scores = self.load_data(data=data, dataset_subjs=val_subj)
        self.build()

        # Size of the datasets used for this GAT model
        tr_size, vl_size = len(train_subj), len(val_subj)
        print('The training size is %d, while the validation one: %d' % (tr_size, vl_size))
        # keep track of best val loss and the last k training losses for early-stopping the training
        tr_k_logs = np.zeros(self.params['k_strip_epochs'])
        global best_vl_loss
        best_vl_loss = np.inf

        def start_time_epoch(epoch, logs):
            global total_time_start
            total_time_start = time.time()

        # eraly stop the training based on generalization loss vs training progress
        def early_stopping(epoch, logs):
            global best_vl_loss
            # pop last tr loss and push the current one
            best_vl_loss = min(best_vl_loss, logs['val_loss'])
            tr_k_logs[:-1] = tr_k_logs[1:]
            tr_k_logs[-1] = logs['loss']
            if 0.0 not in tr_k_logs:
                best_k_tr_loss = np.min(tr_k_logs)
                last_k_tr_loss = np.sum(tr_k_logs)
                gl = (logs['val_loss'] / best_vl_loss - 1.0) * 100.0
                pk = (last_k_tr_loss / (self.params['k_strip_epochs'] * best_k_tr_loss) - 1.0) * 1e3
                print('PQ ratio for epoch is %.6f and the training progress %.5f' % (gl / pk, pk))
                if gl / pk >= self.params['gl_tr_prog_threshold']:
                    self.model.stop_training = True
                if pk <= self.params['enough_train_prog']:
                    self.model.stop_training = True
            global total_time_start
            epoch_time = time.time() - total_time_start
            print('Training: loss = %.5f | Val: loss = %.5f | Elapsed epoch time: %.5f seconds | Epoch number: %d' % (
                logs['loss'], logs['val_loss'], epoch_time * 10, epoch))

        early_stop_callback = LambdaCallback(on_batch_begin=start_time_epoch, on_epoch_end=early_stopping)
        history = self.model.fit(x=[tr_feats, tr_adjs, tr_biases], y=tr_scores,
                                 batch_size=self.params['batch_size'],
                                 epochs=self.params['num_epochs'],
                                 verbose=0,
                                 callbacks=[early_stop_callback],
                                 validation_split=0.0,
                                 validation_data=([vl_feats, vl_adjs, vl_biases], vl_scores),
                                 shuffle=True,
                                 class_weight=None,
                                 sample_weight=None,
                                 initial_epoch=0,
                                 steps_per_epoch=None,
                                 validation_steps=None)
        self.model.save(self.config.checkpt_file())
        with open(self.config.logs_file(), 'wb') as logs_binary:
            pickle.dump({'history': history.history, 'params': self.config.params}, logs_binary)
        self.is_trained = True

    def test(self, data, test_subj):
        if not self.is_trained:
            print('The GAT model %s was not trained yet' % self.config)
            return
        ts_feats, ts_adjs, ts_biases, ts_scores = self.load_data(data=data, dataset_subjs=test_subj)
        ts_size = len(test_subj)
        print('The size of the evaluation set is %d' % ts_size)
        test_loss = self.model.evaluate(x=[ts_feats, ts_adjs, ts_biases], y=ts_scores, batch_size=ts_size,
                                        verbose=0,
                                        sample_weight=None, steps=None)
        with open(self.config.logs_file(), 'rb') as logs_binary:
            logs = pickle.load(logs_binary)
            logs['test_loss'] = test_loss
        with open(self.config.logs_file(), 'wb') as logs_binary:
            pickle.dump(logs, logs_binary)
        print('The test loss for the mode %s is %.2f ' % (self.config, test_loss))
        print()
        clear_session()
