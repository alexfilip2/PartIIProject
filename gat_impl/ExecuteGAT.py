import time
import pickle
import os
from gat_impl.TensorflowGraphGAT import *
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.backend import clear_session
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


class GATModel(TensorflowGraphGAT):

    def __init__(self, config):
        # Load the GAT architecture configuration object of the current model
        if config is None:
            raise TypeError('No GAT configuration object specified')
        else:
            self.config = config
        # Load the hyper-parameter configuration of the current model
        self.params = self.config.params
        # Print the model details
        self.config.print_model_details()
        # Data loading fields
        self.data = None
        self.N = 0
        self.F = 0
        # wrapped Keras model and its status flags
        self.model = None
        self.is_built = False
        self.is_trained = False

    class CustomEarlyStopping(Callback):
        def __init__(self, k_strip_epochs, pq_threshold, train_prog_threshold, **kwargs):
            super(Callback, self).__init__()
            self.best_vl_loss = np.inf
            self.total_time_start = 0.0
            self.k_strip_epochs = k_strip_epochs
            self.pq_threshold = pq_threshold
            self.train_prog_threshold = train_prog_threshold
            self.tr_k_logs = np.zeros(self.k_strip_epochs)
            self._data = {'pq_ratio': [],
                          'train_prog': []}

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
                if gl / pk >= self.pq_threshold:
                    self.model.stop_training = True
                if pk <= self.train_prog_threshold:
                    self.model.stop_training = True
                self._data['pq_ratio'].append(gl / pk)
                self._data['train_prog'].append(pk)
            epoch_time = time.time() - self.total_time_start
            print(
                'Training: loss = %.5f | Val: loss = %.5f | PQ ratio: %.5f | Train progress: %.5f | Elapsed epoch time:'
                ' %.5f seconds | Epoch number: %d' % (logs['loss'], logs['val_loss'], gl / pk, pk, epoch_time, epoch))

        def get_stopping_data(self):
            return self._data

    def build(self):
        # input tensors dimensionality
        feed_data = {'dim_nodes': self.N,
                     'dim_feats': self.F}
        # parameters and inputs for building the graph
        inference_args = {**feed_data, **self.params}

        # Keras GAT model and custom loss function with robustness regularization
        self.model = TensorflowGraphGAT.inference_keras(**inference_args)
        # plot_model(self.model, 'gat_model.pdf', show_shapes=True)

        # Define the optimizer for the training of the model
        optimizer = Adam(lr=self.params['learning_rate'])
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.is_built = True

    def fit(self, training_data, validation_data):
        # load the training data before building the model as it requires the dimensionality
        tr_feats, tr_adjs, tr_biases, tr_scores = training_data
        vl_feats, vl_adjs, vl_biases, vl_scores = validation_data

        # the number of nodes N per example graph
        self.N = tr_adjs.shape[-1]
        # the initial dimension F of each node's feature vector
        self.F = tr_feats.shape[-1]

        # build the architecture
        if not self.is_built:
            self.build()
        # if the model is already trained and persisted on disk, load it weights
        if os.path.exists(self.config.checkpoint_file()):
            self.model.load_weights(self.config.checkpoint_file())
            self.is_trained = True
            return

        # Size of the datasets used by this GAT model
        tr_size, vl_size = len(tr_feats), len(vl_feats)
        print('The training size is %d, while the validation one: %d for the GAT model %s' % (
            tr_size, vl_size, self.config))
        # define the custom early stopping callback
        custom_early_stop = self.CustomEarlyStopping(**self.params)
        # fit the Keras model with the provided data
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
        # save the model weights
        self.model.save_weights(self.config.checkpoint_file())
        # save its training history, early stopping logs along with the hyper-parameters configuration
        with open(self.config.logs_file(), 'wb') as logs_binary:
            pickle.dump({'history': history.history,
                         'early_stop': custom_early_stop.get_stopping_data(),
                         'params': self.config.params}, logs_binary)
        self.is_trained = True

    def save_results(self, predicted, observed):
        predictions = np.transpose(predicted)
        ts_scores = np.transpose(observed)
        # save the results and losses of the evaluation on disk
        with open(self.config.results_file(), 'wb') as results_binary:
            results = {'predictions': {},
                       'test_loss': {},
                       'r2_score': {},
                       'pearson': {},
                       'params': self.config.params}
            for index, pers_trait in enumerate(self.config.params['pers_traits_selection']):
                results['r2_score'][pers_trait] = r2_score(ts_scores[index], predictions[index])
                results['pearson'][pers_trait] = pearsonr(ts_scores[index], predictions[index])
                results['predictions'][pers_trait] = list(zip(ts_scores[index], predictions[index]))
                results['test_loss'][pers_trait] = mean_squared_error(y_true=ts_scores[index],
                                                                      y_pred=predictions[index])
                print('The test loss for trait %s is  %.5f:' % (pers_trait, results['test_loss'][pers_trait]))
            pickle.dump(results, results_binary)
        return results

    def evaluate(self, test_data):
        if not self.is_trained:
            print('The GAT model %s was not trained yet' % self.config)
            return
        ts_feats, ts_adjs, ts_biases, ts_scores = test_data
        ts_size = len(ts_feats)
        print('The size of the evaluation set is %d' % ts_size)
        # predict the scores for the evaluation graphs
        predictions = self.model.predict(x=[ts_feats, ts_adjs, ts_biases],
                                         batch_size=ts_size,
                                         verbose=0,
                                         steps=None)
        # clear the memory of this GAT model
        self.delete()
        # calculate the MSE for individual traits even if they were predicted all at once
        return self.save_results(predicted=predictions, observed=ts_scores)

    def delete(self):
        # clear the main memory of the TensorFlow graph
        del self.model
        clear_session()
