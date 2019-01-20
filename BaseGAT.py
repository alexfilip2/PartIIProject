import tensorflow as tf


class BaseGAT:

    def training(loss, lr, l2_coef):
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
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        # Create optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        # Training tensorflow Operation for batches of size 1
        train_op = opt.minimize(loss + lossL2)

        return train_op

    def batch_training(self, loss, u_loss, e_loss, lr, l2_coef):
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
                    zero_grads_ops : tf Operation
                        The tf operation clearing the gradients accumulated so far for each Trainable Variable
                    accum_ops :  tf Operation
                        The operation accumulating gradients for each Variable instead of using them for update
                    apply_ops : tf Operation
                        The operation  applying the gradients as updates to the Variables after being accumulated
                        for a number of training examples (a batch)
            """
        # create optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        # minibatch operations
        # 0) Retrieve trainable variables
        tvs = tf.trainable_variables()
        # 1) Create placeholders for the accumulating gradients we'll be storing
        accum_vars = [tf.Variable(tv.initialized_value(), trainable=False) for tv in tvs]
        # 2) Operation to initialize accum_vars to zero (reinitialize the gradients)
        zero_grads_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        # 3) Operation to compute the gradients for one minibatch
        # regularization loss of the parameters, exclusive and uniform losses for the robustness of attention network
        l2_l = tf.add_n([tf.nn.l2_loss(v) for v in tvs if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        gvs = opt.compute_gradients(tf.add_n([loss, l2_l, u_loss, e_loss]))
        # 4) Operation to accumulate the gradients in accum_vars
        accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
        # 5) Operation to perform the update (apply gradients)
        apply_ops = opt.apply_gradients([(accum_vars[i], tv) for i, tv in enumerate(tf.trainable_variables())])

        return zero_grads_ops, accum_ops, apply_ops
