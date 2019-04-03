from gat_impl.TensorflowGraphGAT import *


class MainGAT(TensorflowGraphGAT):

    def training(self, loss, u_loss, e_loss, learning_rate, l2_coefficient, **kwargs):
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
        # Returns all variables created which have trainable = True (by default or implicitly)
        vars = tf.trainable_variables()
        # Regularization loss of the parameters: weight decay of the learnable parameters
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coefficient
        # Create optimizer
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # Training tensorflow Operation for batches of size 1
        train_op = opt.minimize(tf.add_n([loss, lossL2, u_loss * l2_coefficient, e_loss * l2_coefficient]))

        return train_op
