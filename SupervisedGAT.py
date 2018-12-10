import time
import numpy as np
import tensorflow as tf

from MainGAT import MainGAT
from Tools import *

checkpt_file = '../PartIIProject/model_base_GAT.ckpt'
if not os.path.exists(checkpt_file):
    check_point = open(checkpt_file,'w+')
    check_point.close()
dataset = 'HCP'

# training params
batch_size = 1  # how many example graphs can be feed in the training process at once
nb_epochs = 1000
patience = 100
lr = 0.0001  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [4, 4, 10]  # numbers of features produced by each attention head per network layer
n_heads = [4, 4, 6]  # number of attention heads on each layer
residual = False
nonlinearity = tf.nn.elu
model = MainGAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj_matrices, graphs_features, score_train, score_test, score_val = load_data()

# nr of nodes for the training graph (transductive learning)nb_nodes = features.shape[0]
nb_nodes = adj_matrices[0].shape[0]
# the F length of each node feature vector specified also in the paper
ft_size = graphs_features.shape[-1]  # for each graph in the dataset the node feat.vecs. have the same length: F
batch_sz = 1
# adjancy matrix for the graph of shape nb_nodes X nb_nodes
# np.newaxis adds a new dimension to the given array (used in order to make compatible with the inductive learning input format
# is a nr_of_graphs X number_of_nodes X feature_vector_size array, the array of all node feat. vec.

# adj = adj[np.newaxis] # shape for adj is now (1, nr_of_nodes, featurevec_size)

biases = adj_to_bias(adj_matrices, [g.shape[0] for g in adj_matrices], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, nb_nodes, nb_nodes))
        score_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, 5))
        adj_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, nb_nodes, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    prediction = model.inference(in_feat_vects=ftr_in,
                                 train_flag=is_train,
                                 attn_drop=attn_drop,
                                 ffd_drop=ffd_drop,
                                 bias_mat=bias_in,
                                 adj_mat=adj_in,
                                 hid_units=hid_units,
                                 n_heads=n_heads,
                                 residual=residual,
                                 activation=nonlinearity)

    loss = tf.losses.mean_squared_error(labels=score_in, predictions=prediction)
    train_op = model.training(loss, lr, l2_coef)
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # a Session object creates the environment in which Operations objects are executed and Tensor objects are evaluated
    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        val_loss_avg = 0

        # nb_epochs gives the number of epochs for training: the number of iteration of gradient descent to optimize the parameters
        for epoch in range(nb_epochs):

            # number of training graph examples
            tr_size = score_train.shape[0]
            # shuffle the training dataset
            shuffled_data = list(zip(score_train, graphs_features[:tr_size], biases[:tr_size], adj_matrices[:tr_size]))
            random.shuffle(shuffled_data)
            score_in_tr, ftr_in_tr, bias_in_tr, adj_in_tr = zip(*shuffled_data)
            assert len(score_in_tr) == len(ftr_in_tr) == len(bias_in_tr) == len(adj_in_tr) == tr_size

            # training step
            tr_step = 0
            # training loss of this epoch
            train_loss_avg = 0

            while tr_step < tr_size:
                (_, loss_value_tr) = sess.run([train_op, loss],
                                              feed_dict={
                                                  ftr_in: ftr_in_tr[tr_step:tr_step + 1],
                                                  bias_in: bias_in_tr[tr_step:tr_step + 1],
                                                  score_in: score_in_tr[tr_step:tr_step + 1],
                                                  adj_in: adj_in_tr[tr_step:tr_step + 1],
                                                  is_train: True,
                                                  attn_drop: 0.6,
                                                  ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                tr_step += 1

            vl_step = tr_size
            vl_size = score_val.shape[0]
            val_loss_avg = 0

            while vl_step < vl_size + tr_size:
                (loss_value_vl,) = sess.run([loss],
                                            feed_dict={
                                                ftr_in: graphs_features[vl_step:vl_step + 1],
                                                bias_in: biases[vl_step:vl_step + 1],
                                                score_in: score_val[vl_step - tr_size:vl_step + 1 - tr_size],
                                                adj_in: adj_matrices[vl_step:vl_step + 1],
                                                is_train: False,
                                                attn_drop: 0.0,
                                                ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                vl_step += 1

            print('Training: loss = %.5f | Val: loss = %.5f' % (train_loss_avg / tr_size, val_loss_avg / vl_size))

        saver.restore(sess, checkpt_file)

        ts_size = score_test.shape[0]
        ts_step = vl_size + tr_size
        ts_loss = 0.0

        while ts_step < ts_size + vl_size + tr_size:
            loss_value_ts = sess.run([loss],
                                     feed_dict={
                                         ftr_in: graphs_features[ts_step:ts_step + 1],
                                         bias_in: biases[ts_step:ts_step + 1],
                                         score_in: score_test[
                                                   ts_step - (vl_size + tr_size): ts_step + 1 - (vl_size + tr_size)],
                                         adj_in: adj_matrices[ts_step:ts_step + 1],
                                         is_train: False,
                                         attn_drop: 0.0,
                                         ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_size)

        sess.close()
