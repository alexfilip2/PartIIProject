from gat_impl.TensorflowGraphGAT import *
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras.backend import clear_session
from sklearn.metrics import mean_squared_error
import time
import pickle
import os


class GATModel(TensorflowGraphGAT):

    def __init__(self, config):
        '''
         Initialize the GAT model object
        :param config: hyperparameters configuration object for the model
        '''
        # load the GAT architecture configuration object of the current model
        if config is None:
            raise TypeError('No GAT configuration object specified')
        else:
            self.config = config
        # load the hyper-parameter configuration of the current model
        self.params = self.config.params
        # Print the model details
        self.config.print_model_details()
        # wrapped Keras GAT model and its status flags
        self.model = None
        self.is_built = False
        self.is_trained = False

    class CustomEarlyStopping(Callback):
        def __init__(self, k_strip_epochs, pq_threshold, train_prog_threshold, **kwargs):
            '''
             Initialize the Callback object for PQ early stopping
            :param k_strip_epochs: number of previous consecutive epochs to consider for the training progress
            :param pq_threshold: threshold for the PQ ration dictating when to stop
            :param train_prog_threshold: threshold for stopping the training when there is no training progress
            :param kwargs: accept dictionary of any actual parameters to instantiate the formal ones
            '''
            super(Callback, self).__init__()
            # time the training time per epoch
            self.total_time_start = 0.0
            # initialize the parameters of the stopping
            self.best_vl_loss = np.inf
            self.k_strip_epochs = k_strip_epochs
            self.pq_threshold = pq_threshold
            self.train_prog_threshold = train_prog_threshold
            self.tr_k_logs = np.zeros(self.k_strip_epochs)
            # store the history of the stopping criterion
            self._data = {'pq_ratio': [], 'train_prog': []}

        def on_epoch_begin(self, epoch, logs=None):
            '''
             Lambda function to apply at the beginning of each epoch, here it is just resetting the timer
            :param epoch: epoch number
            :param logs: the logs of training and validation losses and other metrics
            :return: void
            '''
            self.total_time_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            '''
             Lambda function to apply at the end of each epoch, here it computes the training and generalization
             progress from the metrics obtained on this particular epoch and stops the training if the conditions are
             met. It also pretty prints the metric values of the epoch.
            :param epoch: epoch number
            :param logs: the logs of training and validation losses and other metrics
            :return: void
            '''
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
            '''
             Retrieves the stopping history of the training phase.
            :return: dict of training progress and PQ ratio
            '''
            return self._data

    def build(self, features_shape: tuple):
        '''
         Build the GAT architecture and define the optimizer.
        :return: void
        '''
        # Allocate as much memory as needed and at most a pre-decided fraction of the GPU total memory
        tensorflow_config = tf.ConfigProto()
        tensorflow_config.gpu_options.allow_growth = True
        tensorflow_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # Create a session with the particular options.
        K.set_session(tf.Session(config=tensorflow_config))

        # input tensors dimensionality
        feed_data = {'dim_nodes': features_shape[0],
                     'dim_feats': features_shape[1]}
        # parameters and inputs for building the Keras model
        inference_args = {**feed_data, **self.params}

        # Keras GAT model
        self.model = TensorflowGraphGAT.inference_keras(**inference_args)
        # plot_model(self.model, 'gat_model.pdf', show_shapes=True)

        # Define the optimizer for the training of the model and compile it for optimizing on a loss function
        optimizer = Adam(lr=self.params['learning_rate'])
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.is_built = True

    def fit(self, training_data):
        '''
         Trains the build GAT model.
        :param training_data: the training data of node features, adjacency matrices and masks and score labels
        :param validation_data: the validation data of node features, adjacency matrices and masks and score labels
        :return:
        '''
        # load the data as building the model requires the graph order and node features dimensions beforehand
        tr_feats, tr_adjs, tr_biases = training_data['ftr_in'], training_data['adj_in'], training_data['bias_in']
        tr_scores = training_data['score_in']

        # the graph order of example graphs
        N = tr_adjs.shape[-1]
        # the initial node features dimension
        F = tr_feats.shape[-1]
        # build the architecture
        if not self.is_built:
            self.build(features_shape=(N, F))

        # if the model is already trained and its parameters saved on disk, load them into the skeleton
        if os.path.exists(self.config.checkpoint_file()):
            self.model.load_weights(self.config.checkpoint_file())
            self.is_trained = True
            return
        # report the number training examples
        print('The training size is %d for the GAT model %s' % (len(tr_feats), self.config))

        # define the custom early stopping callback
        custom_early_stop = self.CustomEarlyStopping(**self.params)
        es = EarlyStopping(monitor='val_loss', patience=5)
        # fit the Keras model with the provided data
        history = self.model.fit(x=[tr_feats, tr_adjs, tr_biases],
                                 y=tr_scores,
                                 batch_size=self.params['batch_size'],
                                 epochs=self.params['num_epochs'],
                                 verbose=0,
                                 callbacks=[custom_early_stop, es],
                                 validation_split=0.2,
                                 validation_data=None,
                                 shuffle=True,
                                 class_weight=None,
                                 sample_weight=None,
                                 initial_epoch=0,
                                 steps_per_epoch=None,
                                 validation_steps=None)
        # save the model weights after training
        self.model.save_weights(self.config.checkpoint_file())
        # save the training history, early stopping logs along with the hyper-parameters configuration
        with open(self.config.logs_file(), 'wb') as logs_binary:
            pickle.dump({'history': history.history,
                         'early_stop': custom_early_stop.get_stopping_data(),
                         'params': self.config.params}, logs_binary)
        self.is_trained = True

    def save_results(self, predicted, observed):
        '''
         Save the results of the evaluation of the model: predictions and real scores, the loss on predicting each
         individual trait along with the hyperparameters used
        :param predicted: array of predictions
        :param observed: array of true scores
        :return: dict of these result values
        '''
        # stack the results per trait
        predictions = np.transpose(predicted)
        ts_scores = np.transpose(observed)
        # save the results and losses of the evaluation on disk along with the hyper-parameter configuration
        with open(self.config.results_file(), 'wb') as results_binary:
            results = {'predictions': {},
                       'test_loss': {},
                       'params': self.config.params}
            for index, pers_trait in enumerate(self.config.params['pers_traits_selection']):
                results['predictions'][pers_trait] = list(zip(ts_scores[index], predictions[index]))
                results['test_loss'][pers_trait] = mean_squared_error(y_true=ts_scores[index],
                                                                      y_pred=predictions[index])
                print('The test loss for trait %s is  %.2f:' % (pers_trait, results['test_loss'][pers_trait]))
            pickle.dump(results, results_binary)
        return results

    def evaluate(self, test_data):
        '''
         Evaluate the trained GAT model.
        :param test_data: the test data of node features, adjacency matrices and masks and score labels
        :return: dict of the evaluation results: metric values and actual predictions
        '''
        if not self.is_trained:
            print('The GAT model %s was not trained yet' % self.config)
            return
        ts_feats, ts_adjs, ts_biases = test_data['ftr_in'], test_data['adj_in'], test_data['bias_in']
        ts_scores = test_data['score_in']
        ts_size = len(ts_feats)
        print('The size of the evaluation set is %d' % ts_size)
        # predict the scores for the evaluation graphs
        predictions = self.model.predict(x=[ts_feats, ts_adjs, ts_biases],
                                         batch_size=ts_size,
                                         verbose=0,
                                         steps=None)
        # clear the memory of the Keras model
        self.delete()
        # calculate the MSE for individual traits even if they were predicted all at once
        return self.save_results(predicted=predictions, observed=ts_scores)

    def delete(self):
        '''
         Clear the main memory/ GPU memory of the Keras model and Tensorflow default session
        :return: void
        '''
        # delete the main memory of the GAT model
        del self.model
        # clear the Tensorflow default session
        clear_session()
