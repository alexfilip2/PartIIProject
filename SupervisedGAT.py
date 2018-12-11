import time
import numpy as np
import tensorflow as tf

from MainGAT import MainGAT
from Tools import *
from ToolsStructural import *

checkpt_file = '../PartIIProject/model_base_GAT_noEdgeWeights.ckpt'

dataset = 'HCP'

# training params
batch_size = 1  # how many example graphs can be feed in the training process at once
nb_epochs = 1000
patience = 100
lr = 0.0001  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [10, 15, 15]  # numbers of features produced by each attention head per network layer
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

# personality traits scores: 'NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E'
trait_choice = ['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E']
# data for adjancency matrices, node feature vectors and personality scores for each study patient
adj_matrices, graphs_features, score_train, score_test, score_val = load_struct_data(trait_choice)
# used in order to implement MASKED ATTENTION by discardining non-neighbours out of nhood hops
biases = adj_to_bias(adj_matrices, [graph.shape[0] for graph in adj_matrices], nhood=1)
# nr of nodes for each graph: it is shared among all examples due to the dataset
nb_nodes = adj_matrices[0].shape[0]
# the initial lenagth F of each node feature vector: for every graph, node feat.vecs. have the same length
ft_size = graphs_features.shape[-1]
# batch training size; currently ONLY ONE example per training step: TO BE EXTENDED!!
batch_sz = 1
# how many of the big-five personality traits the model is targeting at once
outGAT_sz_target = len(trait_choice)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, nb_nodes, nb_nodes))
        score_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, outGAT_sz_target))
        adj_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, nb_nodes, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    prediction = model.inference(in_feat_vects=ftr_in,
                                 adj_mat=adj_in,
                                 bias_mat=bias_in,
                                 hid_units=hid_units,
                                 n_heads=n_heads,
                                 target_score_type=outGAT_sz_target,
                                 train_flag=is_train,
                                 residual=residual,
                                 activation=nonlinearity,
                                 attn_drop=attn_drop,
                                 ffd_drop=ffd_drop)

    loss = tf.losses.mean_squared_error(labels=score_in, predictions=prediction)
    train_op = model.training(loss, lr, l2_coef)
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # a Session object creates the environment in which Operations objects are executed and Tensor objects are evaluated
    with tf.Session() as sess:
        # number of training graph examples
        tr_size = len(score_train)
        # number of validation graph examples
        vl_size = len(score_val)
        # number of test graph examples
        ts_size = len(score_test)
        print('The training size is: %d, the validation: %d and the test: %d' % (tr_size, vl_size, ts_size))

        if not tf.train.checkpoint_exists(checkpt_file):
            print('The GAT model is not pre-trained, training it now...')
            # run the initializer for the Variable objects to be optimized
            sess.run(init_op)
            vars = tf.all_variables()
            print('The learnable parametres of the GAT model are:')
            for var in vars: print(var.name)

            # define the losses of the training process
            train_loss_avg = 0
            val_loss_avg = 0

            # nb_epochs - number of epochs for training: the number of iteration of gradient descent to optimize
            for epoch in range(nb_epochs):

                # shuffle the training dataset
                score_in_tr, ftr_in_tr, bias_in_tr, adj_in_tr = shuffle_tr_data(score_train,
                                                                                graphs_features,
                                                                                biases,
                                                                                adj_matrices,
                                                                                tr_size)

                # training step
                tr_step = 0
                #  training loss of this epoch
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

                # validation step
                vl_step = tr_size
                # validation loss of this epoch
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

            save_path = saver.save(sess, checkpt_file)
            print("Model saved in path: %s" % save_path)

        saver.restore(sess, checkpt_file)
        print("Model restored.")

        ts_step = tr_size + vl_size
        ts_loss = 0.0

        while ts_step < ts_size + len(score_train) + len(score_val):
            (loss_value_ts,) = sess.run([loss],
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
