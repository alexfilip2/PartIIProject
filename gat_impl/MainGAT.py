from gat_impl.TensorflowGraphGAT import *


class MainGAT(TensorflowGraphGAT):

    def training(self, loss, u_loss, e_loss, learning_rate, decay_rate, **kwargs):
        """ Defines the training operation of the entire GAT neural network model

                Parameters
                ----------
                loss : function
                    The loss function of the regression, it can be MSE, MAE, etc
                lr : float
                    The learning rate of the parameters
                l2_coef : float
                    L2 regularization coefficient

                Returns
                -------
                train_op : tf Operation
                    The training operation using the Adam optimizer to learn the parameters by minimizing 'loss'
        """
        # Create optimizer
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # Training tensorflow Operation for batches of size 1
        train_op = opt.minimize(tf.add_n([loss, u_loss * decay_rate, e_loss * decay_rate]))

        return train_op
