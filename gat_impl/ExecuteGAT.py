import time
import pickle
from gat_impl.MainGAT import *
from gat_impl.HyperparametersGAT import HyperparametersGAT
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, EarlyStopping, Callback
from keras.backend import clear_session
from sklearn.metrics import mean_squared_error

import os


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
        self.is_trained = False

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

    class CustomEarlyStopping(Callback):
        def __init__(self, **kwargs):
            super(Callback, self).__init__()
            self.best_vl_loss = np.inf
            self.total_time_start = 0.0
            self.k_strip_epochs = kwargs['k_strip_epochs']
            self.gl_tr_prog_threshold = kwargs['gl_tr_prog_threshold'][kwargs['readout_aggregator']]
            self.enough_train_prog = kwargs['enough_train_prog']
            self.tr_k_logs = np.zeros(self.k_strip_epochs)
            self._data = {'pq_ratio': [], 'train_prog': []}

        def on_epoch_begin(self, epoch, logs=None):
            self.total_time_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            # pop last tr loss and push the current one
            self.best_vl_loss = min(self.best_vl_loss, logs['val_loss'])
            self.tr_k_logs[:-1] = self.tr_k_logs[1:]
            self.tr_k_logs[-1] = logs['loss']
            gl = 0.0
            pk = np.inf
            if 0.0 not in self.tr_k_logs:
                best_k_tr_loss = np.min(self.tr_k_logs)
                last_k_tr_loss = np.sum(self.tr_k_logs)
                gl = (logs['val_loss'] / self.best_vl_loss - 1.0) * 100.0
                pk = (last_k_tr_loss / (self.k_strip_epochs * best_k_tr_loss) - 1.0) * 1e3
                if gl / pk >= self.gl_tr_prog_threshold:
                    self.model.stop_training = True
                if pk <= self.enough_train_prog:
                    self.model.stop_training = True
                self._data['pq_ratio'].append(gl / pk)
                self._data['train_prog'].append(pk)
            epoch_time = time.time() - self.total_time_start
            print(
                'Training: loss = %.5f | Val: loss = %.5f | PQ ratio: %.5f | Train progress: %.5f | Elapsed epoch time:'
                ' %.5f seconds | Epoch number: %d' % (
                    logs['loss'], logs['val_loss'], gl / pk, pk, epoch_time, epoch))

        def get_data(self):
            return self._data

    def build(self):
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
        if os.path.exists(self.config.checkpt_file()):
            self.model.load_weights(self.config.checkpt_file())
            self.is_trained = True
            return

        # Size of the datasets used for this GAT model
        tr_size, vl_size = len(train_subj), len(val_subj)
        print('The training size is %d, while the validation one: %d' % (tr_size, vl_size))
        custom_early_stop = self.CustomEarlyStopping(**self.params)
        es = EarlyStopping(monitor='val_loss', mode='min', patience=15)

        history = self.model.fit(x=[tr_feats, tr_adjs, tr_biases], y=tr_scores,
                                 batch_size=self.params['batch_size'],
                                 epochs=self.params['num_epochs'],
                                 verbose=0,
                                 callbacks=[custom_early_stop],
                                 validation_split=0.0,
                                 validation_data=([vl_feats, vl_adjs, vl_biases], vl_scores),
                                 shuffle=True,
                                 class_weight=None,
                                 sample_weight=None,
                                 initial_epoch=0,
                                 steps_per_epoch=None,
                                 validation_steps=None)
        # save the model weights, its training history and hyper-parameters configuration
        self.model.save_weights(self.config.checkpt_file())
        with open(self.config.logs_file(), 'wb') as logs_binary:
            pickle.dump({'history': history.history,
                         'early_stop': custom_early_stop.get_data(),
                         'params': self.config.params}, logs_binary)
        self.is_trained = True

    def test(self, data, test_subj):
        if not self.is_trained:
            print('The GAT model %s was not trained yet' % self.config)
            return
        ts_feats, ts_adjs, ts_biases, ts_scores = self.load_data(data=data, dataset_subjs=test_subj)
        ts_size = len(test_subj)
        print('The size of the evaluation set is %d' % ts_size)
        predictions = self.model.predict(x=[ts_feats, ts_adjs, ts_biases],
                                         batch_size=ts_size,
                                         verbose=0,
                                         steps=None)

        predictions = np.transpose(predictions)
        ts_scores = np.transpose(ts_scores)
        with open(self.config.logs_file(), 'rb') as logs_binary:
            logs = pickle.load(logs_binary)
            logs['test_loss'] = {}
            for index, pers_trait in enumerate(self.config.params['pers_traits_selection']):
                logs['test_loss'][pers_trait] = mean_squared_error(y_true=ts_scores[index], y_pred=predictions[index])
                print('The test loss for trait %s is  %.5f:' % (pers_trait, logs['test_loss'][pers_trait]))

        with open(self.config.logs_file(), 'wb') as logs_binary:
            pickle.dump(logs, logs_binary)

        clear_session()
