import tensorflow as tf


class BaseGAT:

    def training(loss, lr, l2_coef):
        """ Defines the training operation of the whole GAT neural network model

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
                train_op : function
                    The training operation using the Adam optimizer to learn the parameters by minimizing 'loss'
                """
        # weight decay
        # Returns all variables created which have trainable = True (by default or implicitly)
        vars = tf.trainable_variables()
        # regularization loss of the parameters
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        # training op
        train_op = opt.minimize(loss + lossL2)

        return train_op
