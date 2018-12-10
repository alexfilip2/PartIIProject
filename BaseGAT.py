import tensorflow as tf


class BaseGAT:
    '''
    labels - an array in which each entry in labels must be an index in [0, nb_classes)
    logits - arrays of length nb_classes, unnormalized, results of MainGAT.inference
    nb_classes - number of classes
    '''
    def training(loss, lr, l2_coef):
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



    ##########################
    # Adapted from tkipf/gcn #
    ##########################

